import numpy as np
import math
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product
from quantizers.uniform import UniformQuantizer
from quantizers.logarithm import Log2Quantizer, LogSqrt2Quantizer, AdaLogQuantizer
from datetime import datetime


class MinMaxQuantMatMul(nn.Module):
    def __init__(self, 
                 A_bit = 8, 
                 B_bit = 8, 
                 mode = "raw"):
        super().__init__()
        self.mode = mode
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = False)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = False)
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
    
    def forward(self, A, B):
        if self.mode == 'raw':
            out = A @ B
        elif self.mode == "quant_forward":
            out = self.quant_forward(A, B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input_A(self, x):
        return self.A_quantizer(x)
    
    def quant_input_B(self, x):
        return self.B_quantizer(x)
    
    def quant_forward(self, A, B):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        return self.quant_input_A(A) @ self.quant_input_B(B)
    
    
class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    - Q @ K:
        - A's shape: B,H,S,C
        - B's shape: B,H,C,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,C
    """
    def __init__(self, 
                 A_bit = 8, 
                 B_bit = 8, 
                 mode = "raw", 
                 search_round = 1, 
                 eq_n = 100, 
                 head_channel_wise = True, 
                 num_heads = 12):
        super().__init__(A_bit, B_bit, mode)
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = True, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = True, channel_wise = head_channel_wise)
        self.search_round = search_round
        self.eq_n = eq_n
        # the head dim is always dim-1
        self.head_channel_wise = head_channel_wise
        self.num_heads = num_heads
        
        target_shape = [1, self.num_heads, 1, 1] if self.head_channel_wise else [1, 1, 1, 1]
        self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
    
    def _get_similarity(self, tensor_raw, tensor_sim):
        return -(tensor_raw - tensor_sim) ** 2
        
    
class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, 
                 A_bit = 8, 
                 B_bit = 8, 
                 mode = "raw", 
                 calib_batch_size = 32,
                 search_round = 1, 
                 eq_n = 100, 
                 head_channel_wise = True, 
                 num_heads = 12):
        super().__init__(A_bit, B_bit, mode, search_round, eq_n, head_channel_wise, num_heads)
        self.calib_batch_size = calib_batch_size
        
    def _initialize_calib_parameters(self):
        self.calib_size = self.raw_input[0].shape[0]
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            memory = props.total_memory // 2
        else:
            raise EnvironmentError("CUDA is not available on this system")
        numel = (4 * self.raw_input[0][:self.calib_size].numel()+
                 4 * self.raw_input[1][:self.calib_size].numel()+
                 8 * self.raw_out[:self.calib_batch_size].numel()) # number of parameters on GPU
        self.parallel_eq_n = int((memory / 4) // numel)
        self.parallel_eq_n = math.ceil(self.eq_n * 1.0 / math.ceil(self.eq_n * 1.0 / self.parallel_eq_n))
        
        
class AsymmetricallyBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, 
                 A_bit = 8, 
                 B_bit = 8, 
                 mode = "raw", 
                 calib_batch_size = 32, 
                 search_round = 1, 
                 eq_n = 128, 
                 head_channel_wise = True, 
                 num_heads = 12, 
                 fpcs = False, 
                 steps = 4):
        super().__init__(A_bit, B_bit, mode, calib_batch_size, search_round, eq_n, head_channel_wise, num_heads)
        self.fpcs = fpcs
        self.steps = steps
        
        del self.A_quantizer, self.B_quantizer
        self.A_quantizer = UniformQuantizer(n_bits = A_bit, symmetric = False, channel_wise = head_channel_wise)
        self.B_quantizer = UniformQuantizer(n_bits = B_bit, symmetric = False, channel_wise = head_channel_wise)
        
        target_shape = [1, self.num_heads, 1, 1] if self.head_channel_wise else [1, 1, 1, 1]
        self.A_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.scale = nn.Parameter(torch.zeros(*target_shape))
        self.A_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
        self.B_quantizer.zero_point = nn.Parameter(torch.zeros(*target_shape))
    
    def _search_best_A_scale(self, A_scale_candidates, A_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize A
                cur_A_scale = A_scale_candidates[p_st:p_ed]
                cur_A_zero_point = A_zero_point_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_quant = ((A_sim / cur_A_scale).round_() + cur_A_zero_point).clamp(0, 2 * self.A_quantizer.n_levels - 1)
                A_sim = (A_quant - cur_A_zero_point).mul_(cur_A_scale) # shape: (parallel_eq_n,b,*,dim1,dim2)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, -1, 1, 1)
        if topk == 1:
            tmp_A_scale = torch.gather(A_scale_candidates, dim=0, index=best_index)
            tmp_A_zero_point = torch.gather(A_zero_point_candidates, dim=0, index=best_index)
            self.A_quantizer.scale.data.copy_(tmp_A_scale.view(self.A_quantizer.scale.shape))
            self.A_quantizer.zero_point.copy_(tmp_A_zero_point.view(self.A_quantizer.zero_point.shape))
        return best_index
        
    def _search_best_B_scale(self, B_scale_candidates, B_zero_point_candidates, topk=1):
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            A_sim = self.quant_input_A(A).unsqueeze(0) # shape: 1,b,*,dim1,dim2
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st + self.parallel_eq_n)
                # quantize B
                cur_B_scale = B_scale_candidates[p_st:p_ed]
                cur_B_zero_point = B_zero_point_candidates[p_st:p_ed]
                B_sim = B.squeeze(0)
                B_quant = ((B_sim / cur_B_scale).round_() + cur_B_zero_point).clamp(0, 2 * self.B_quantizer.n_levels - 1)
                B_sim = (B_quant - cur_B_zero_point).mul_(cur_B_scale) # shape: (parallel_eq_n,b,*,dim2,dim3)
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim) # shape: parallel_eq_n,b,*,dim1,dim3
                if self.head_channel_wise:
                    similarity = torch.mean(similarity, dim=list(range(3, len(similarity.shape)))) # shape: parallel_eq_n,b,heads
                else:
                    similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=False) #shape: eq_n or (eq_n,heads)
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, -1, 1, 1)
        if topk == 1:
            tmp_B_scale = torch.gather(B_scale_candidates, dim=0, index=best_index)
            tmp_B_zero_point = torch.gather(B_zero_point_candidates, dim=0, index=best_index)
            self.B_quantizer.scale.data.copy_(tmp_B_scale.view(self.B_quantizer.scale.shape))
            self.B_quantizer.zero_point.copy_(tmp_B_zero_point.view(self.B_quantizer.zero_point.shape))
        return best_index
    
    def calculate_percentile_candidates(self, x, l=0.9, r=1.0):
        num_zp = min(16, self.B_quantizer.n_levels)
        num_scale = int(self.eq_n / num_zp)
        percentiles_uppers, percentiles_lowers = [], []
        pct = torch.tensor([l, r])
        tensor_too_large = True
        mini_batch_size = 1
        if self.head_channel_wise:
            x_ = x.transpose(0, 1).contiguous() # shape: heads,b,*,dim1,dim2
            x_ = x_.view(x_.shape[0], mini_batch_size, -1) 
        else:
            x_ = x.view(1, mini_batch_size, -1)
        while tensor_too_large:
            try:
                uppers_candidates = torch.quantile(x_, pct.to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                lowers_candidates = torch.quantile(x_, (1 - pct).to(x_.device), dim=-1).mean(dim=-1, keepdim=False) # shape: 2,(heads or 1)
                tensor_too_large = False
            except:
                mini_batch_size *= 2
                x_ = x_.view(x_.shape[0], mini_batch_size, -1) if self.head_channel_wise else x_.view(1, mini_batch_size, -1)
        delta_min = (uppers_candidates[0] - lowers_candidates[0]).view(1, 1, -1, 1, 1)
        delta_max = (uppers_candidates[1] - lowers_candidates[1]).view(1, 1, -1, 1, 1)
        splits = torch.linspace(0, 1, steps=num_scale).cuda()[:, None, None, None, None] * (delta_max - delta_min)
        scale_candidates = (delta_min + splits).repeat(num_zp, 1, 1, 1, 1) / (2 * self.B_quantizer.n_levels - 1)
        zp_min = int(self.B_quantizer.n_levels - num_zp / 2)
        zp_max = int(self.B_quantizer.n_levels + num_zp / 2)
        zp_candidates = torch.tensor(range(zp_min, zp_max)).cuda()
        zero_point_candidates = zp_candidates.repeat_interleave(num_scale)[:, None, None, None, None]
        zero_point_candidates = zero_point_candidates.repeat(1, *scale_candidates.shape[1:])
        return scale_candidates, zero_point_candidates
        
    
    def _fpcs(self, x, fpcs_width=16, steps=6, search_strategy=None):
        fpcs_new_cnt = int(self.eq_n / fpcs_width)
        scale_candidates, zero_point_candidates = self.calculate_percentile_candidates(x)
        delta_scale = scale_candidates[1:2] - scale_candidates[0:1]
        topk_index = search_strategy(self, scale_candidates, zero_point_candidates, topk=fpcs_width)
        topk_scale_candidates = torch.gather(scale_candidates, dim=0, index=topk_index)
        topk_zp_candidates = torch.gather(zero_point_candidates, dim=0, index=topk_index)
        remain_steps = steps - 1
        while remain_steps > 0:
            delta_scale_candidates = (torch.linspace(0, 1, steps=fpcs_new_cnt).cuda()[:, None, None, None, None] - 0.5) * delta_scale
            delta_scale = delta_scale / (fpcs_new_cnt - 0.5)
            scale_candidates = (topk_scale_candidates.unsqueeze(1) + delta_scale_candidates.unsqueeze(0)).reshape(
                -1, *scale_candidates.shape[1:])
            zero_point_candidates = topk_zp_candidates.repeat_interleave(fpcs_new_cnt, dim=0)
            topk_index = search_strategy(self, scale_candidates, zero_point_candidates, 
                                         topk=1 if remain_steps == 1 else fpcs_width)
            if remain_steps > 1:
                topk_scale_candidates = torch.gather(scale_candidates, dim=0, index=topk_index)
                topk_zp_candidates = torch.gather(zero_point_candidates, dim=0, index=topk_index)
            remain_steps -= 1
        
    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        A_scale_candidates, A_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[0].cuda())
        B_scale_candidates, B_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda())
        self.A_quantizer.scale.data.copy_(A_scale_candidates[-2])
        self.A_quantizer.zero_point.data.copy_(A_zero_point_candidates[-2])
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.A_quantizer.inited, self.B_quantizer.inited = True, True
        
        for e in range(self.search_round):
            if self.fpcs:
                self._fpcs(self.raw_input[0].cuda(), steps=self.steps, search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_A_scale)
                self._fpcs(self.raw_input[1].cuda(), steps=self.steps, search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_B_scale)
            else:
                self._search_best_A_scale(A_scale_candidates, A_zero_point_candidates)
                self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
        
        
class PostSoftmaxAsymmetricallyBatchingQuantMatMul(AsymmetricallyBatchingQuantMatMul):
    '''
    log_2 base = 1/k
    k = r / q, here we set r = 37, and search the best q.
    '''
    def __init__(self, 
                 A_bit = 8, 
                 B_bit = 8, 
                 mode = "raw", 
                 calib_batch_size = 32, 
                 search_round = 1, 
                 eq_n = 100, 
                 head_channel_wise = True, 
                 num_heads = 12, 
                 fpcs = False, 
                 steps = 4,
                 quantizer='adalog'):
        super().__init__(A_bit, B_bit, mode, calib_batch_size, search_round, 
                         eq_n, head_channel_wise, num_heads, fpcs, steps)
        del self.A_quantizer
        target_shape = [1, 1, 1, 1]
        if quantizer == 'log2':
            self.A_quantizer = Log2Quantizer(n_bits = A_bit, symmetric = False, channel_wise = False)
        elif quantizer == 'logsqrt2':
            self.A_quantizer = LogSqrt2Quantizer(n_bits = A_bit, symmetric = False, channel_wise = False)
        elif quantizer == 'adalog':
            self.A_quantizer = AdaLogQuantizer(n_bits = A_bit, symmetric = False, channel_wise = False)
            self.table = torch.tensor([2 ** (-j/self.A_quantizer.r) for j in range(120)])
            self.table_scale = 1. / (4 * self.A_quantizer.n_levels - 2)
            self.table = torch.round(self.table / self.table_scale) * self.table_scale
        else:
            raise NotImplementedError(f"quantizer {quantizer} not implemented!")
        self.A_quantizer.scale = nn.Parameter(torch.ones(target_shape))
        self.A_quantizer.inited = True
    
    def _search_best_A_log_base(self, q_candidates=None, topk=1):
        if q_candidates is None:
            q_candidates = torch.tensor([i for i in range(10, 11 + self.eq_n)]).cuda().view(-1, 1, 1, 1, 1)
        batch_similarities = [] # similarities, need to concatenate and calculate sum
        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)
            A = self.raw_input[0][b_st:b_ed].cuda()
            B = self.raw_input[1][b_st:b_ed].cuda()
            B_sim = self.quant_input_B(B).unsqueeze(0) # shape: 1,b,*,dim2,dim3
            raw_out = self.raw_out[b_st:b_ed].unsqueeze(0).cuda()
            similarities = []
            for p_st in range(0, self.eq_n, self.parallel_eq_n):
                p_ed = min(self.eq_n, p_st+self.parallel_eq_n)
                # quantize A
                cur_q = q_candidates[p_st:p_ed]
                A_sim = A.squeeze(0)
                A_sim_quant = torch.round(-A_sim.log2() * self.A_quantizer.r / cur_q) # shape: parallel_eq_n, b, heads, *
                mask = A_sim_quant >= 2 * self.A_quantizer.n_levels
                A_sim_quant = A_sim_quant.clamp_(0, 2 * self.A_quantizer.n_levels - 1)
                col_index = torch.remainder(A_sim_quant * cur_q, self.A_quantizer.r).round_().long()
                A_sim = (2 ** (-1 * torch.floor(A_sim_quant * cur_q / self.A_quantizer.r))) * self.table.to(A.device)[col_index]
                A_sim[mask] = 0
                out_sim = A_sim @ B_sim # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = self._get_similarity(raw_out, out_sim) # shape: parallel_eq_n,b,*,dim1,dim3
                similarity = torch.mean(similarity, dim=list(range(2, len(similarity.shape)))) # shape: parallel_eq_n,b
                similarity = similarity.sum(dim=1, keepdim=True) # shape: (parallel_eq_n,1) or (parallel_eq_n,1,heads)
                similarities.append(similarity)
            # calculate best similarity for this block
            similarities = torch.cat(similarities, 0) # shape: (eq_n,1) or (eq_n,1,heads)
            batch_similarities.append(similarities)
        batch_similarities = torch.cat(batch_similarities, dim=1).sum(dim=1, keepdim=True) #shape: eq_n,1
        _, best_index = torch.topk(batch_similarities, k=topk, dim=0)
        best_index = best_index.view(topk, 1, 1, 1, 1) # shape: topk,...
        if topk == 1:
            tmp_q = torch.gather(q_candidates, dim=0, index=best_index)
            self.A_quantizer.q.data.copy_(tmp_q.view(*self.A_quantizer.q.shape))
            self.A_quantizer.update_table()
        return best_index

    def hyperparameter_searching(self):
        self._initialize_calib_parameters()
        B_scale_candidates, B_zero_point_candidates = self.calculate_percentile_candidates(self.raw_input[1].cuda())
        self.B_quantizer.scale.data.copy_(B_scale_candidates[-2])
        self.B_quantizer.zero_point.data.copy_(B_zero_point_candidates[-2])
        self.B_quantizer.inited = True
        
        for e in range(self.search_round):
            if isinstance(self.A_quantizer, AdaLogQuantizer):
                self._search_best_A_log_base()
            if self.fpcs:
                self._fpcs(self.raw_input[1].cuda(), steps=self.steps, search_strategy=AsymmetricallyBatchingQuantMatMul._search_best_B_scale)
            else:
                self._search_best_B_scale(B_scale_candidates, B_zero_point_candidates)
            if not isinstance(self.A_quantizer, AdaLogQuantizer):
                break
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None
