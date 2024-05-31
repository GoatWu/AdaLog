import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import timm
from timm.models.swin_transformer import window_partition, window_reverse
from utils.calibrator import QuantCalibrator
from quantizers.adaround import AdaRoundQuantizer
from quant_layers import *
from types import MethodType
import logging
import random
import copy


class BlockReconstructor(QuantCalibrator):
    def __init__(self, model, full_model, calib_loader):
        super().__init__(model, calib_loader)
        self.full_model = full_model
        self.blocks = {}
        self.full_blocks = {}
        types_of_block = [
            timm.layers.patch_embed.PatchEmbed,
            timm.models.vision_transformer.Block,
            timm.models.swin_transformer.SwinTransformerBlock,
            timm.models.swin_transformer.PatchMerging,
        ]
        for name, module in self.model.named_modules():
            if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'head':
                self.blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)
        for name, module in self.full_model.named_modules():
            if any(isinstance(module, t) for t in types_of_block) or name.split('.')[-1] == 'head':
                self.full_blocks[name] = module
                BlockReconstructor._prepare_module_data_init(module)
                
    @staticmethod
    def _prepare_module_data_init(module):
        module.raw_input = module.tmp_input = None
        module.raw_out = module.tmp_out = None
                
    def set_block_mode(self, block, mode='raw'):
        for _, module in block.named_modules():
            if hasattr(module, 'mode'):
                module.mode = mode
                
    def wrap_quantizers_in_net(self, block, name):
        print('wraping quantizers in {} ...'.format(name))
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                if isinstance(module, MinMaxQuantLinear):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.n_V, module.crb_rows, module.in_features), 
                                                           round_mode='learned_hard_sigmoid')
                elif isinstance(module, MinMaxQuantConv2d):
                    module.w_quantizer = AdaRoundQuantizer(uq = module.w_quantizer, 
                                                           weight_tensor = module.weight.view(module.weight.shape[0], -1), 
                                                           round_mode='learned_hard_sigmoid')
                module.w_quantizer.soft_targets = True

    def init_block_raw_data(self, block, full_block, name, device, keep_gpu=True):
        self.init_block_raw_inp_outp(block, full_block, name, device)
        if keep_gpu:
            block.raw_input, block.raw_out = block.raw_input.to(device), block.raw_out.to(device)

    def init_block_raw_inp_outp(self, block, full_block, name, device):
        logging.info('initializing raw input and raw output ...')
        for _name, _block in self.blocks.items():
            self.set_block_mode(_block, 'raw')
        hooks = []
        hooks.append(full_block.register_forward_hook(self.outp_forward_hook))
        hooks.append(full_block.register_forward_hook(self.single_input_forward_hook))
        with torch.no_grad():
            for inp, target in self.calib_loader:
                inp = inp.to(device)
                _ = self.full_model(inp)
        block.raw_out = torch.cat(full_block.tmp_out, dim=0)
        block.raw_input = torch.cat(full_block.tmp_input, dim=0)
        full_block.tmp_input, full_block.tmp_out = None, None
        for hook in hooks:
            hook.remove()
            
    def reconstruct_single_block(self, name, block, device,
                                 batch_size: int = 32, iters: int = 20000, weight: float = 0.01,
                                 b_range: tuple = (20, 2), warmup: float = 0.2, lr: float = 4e-5, p: float = 2.0, 
                                 quant_act = False):
        self.wrap_quantizers_in_net(block, name)
        self.set_block_mode(block, 'quant_forward')
        for _name, module in block.named_modules():
            if hasattr(module, 'training_mode'):
                module.init_training()
        w_params, a_params = [], []
        bias_params = []
        for _name, module in block.named_modules():
            if hasattr(module, 'mode'):
                if isinstance(module, MinMaxQuantLinear) or isinstance(module, MinMaxQuantConv2d):
                    w_params += [module.w_quantizer.alpha]
                    if quant_act:
                        a_params += [module.a_quantizer.scale]
                    else:
                        module.mode = 'debug_only_quant_weight'
                elif isinstance(module, MinMaxQuantMatMul):
                    if quant_act:
                        a_params += [module.A_quantizer.scale, module.B_quantizer.scale]
                    else:
                        module.mode = 'raw'
        w_optimizer = torch.optim.Adam(w_params)
        a_optimizer = torch.optim.Adam(a_params, lr=lr) if len(a_params) != 0 else None
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_optimizer, T_max=iters, eta_min=0.) if len(a_params) != 0 else None
        loss_func = LossFunction(block, round_loss='relaxation', weight=weight, max_count=iters, 
                                 rec_loss='mse' if 'head' not in name else 'kl_div',
                                 b_range=b_range, decay_start=0, warmup=warmup, p=p)
        for it in range(iters):
            idx = torch.randperm(block.raw_input.size(0))[:batch_size]
            cur_inp = block.raw_input[idx].to(device)
            cur_out = block.raw_out[idx].to(device)
            w_optimizer.zero_grad()
            if quant_act:
                a_optimizer.zero_grad()
            out_quant = block(cur_inp)
            err = loss_func(out_quant, cur_out)
            err.backward()
            w_optimizer.step()
            if quant_act:
                a_optimizer.step()
                a_scheduler.step()
        # Finish optimization, use hard rounding.
        for name, module in block.named_modules():
            if hasattr(module, 'w_quantizer'):
                module.w_quantizer.soft_targets = False
            if hasattr(module, 'mode'):
                module.mode = 'raw'
            if hasattr(module, 'training_mode'):
                module.end_training()
        del block.raw_input, block.raw_out
        torch.cuda.empty_cache()

    def reconstruct_model(self, quant_act: bool = False, keep_gpu: bool = True):
        device = next(self.model.parameters()).device
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'raw'
        for idx, name in enumerate(self.blocks.keys()):
            block, full_block = self.blocks[name], self.full_blocks[name]
            logging.info('reconstructing {} ...'.format(name))
            self.init_block_raw_data(block, full_block, name, device, keep_gpu=keep_gpu)
            logging.info('adaround training for {} ...'.format(name))
            self.reconstruct_single_block(name, block, device, quant_act=quant_act)
            logging.info('finished reconstructing {}.'.format(name))
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = 'quant_forward'
            if hasattr(module, 'w_quantizer'):
                module.weight.data.copy_(module.w_quantizer.get_hard_value(module.weight.data))
                del module.w_quantizer.alpha
                module.w_quantizer.round_mode = "nearest"

        
class LossFunction:
    def __init__(self,
                 block,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
    
    @staticmethod
    def lp_loss(pred, tgt, p=2.0, reduction='none'):
        """
        loss function measured in L_p Norm
        """
        if reduction == 'none':
            return (pred-tgt).abs().pow(p).sum(1).mean()
        else:
            return (pred-tgt).abs().pow(p).mean()

    def __call__(self, pred, tgt):
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = self.lp_loss(pred, tgt, p=self.p) / 10
        elif self.rec_loss == 'kl_div':
            rec_loss = F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(tgt, dim=-1).detach(), reduction="batchmean")
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = round_loss_pow2 = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if hasattr(module, 'w_quantizer'):
                    round_vals = module.w_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count == 1 or self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
