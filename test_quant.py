import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
import time

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from datetime import datetime
import logging

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoints/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/vit_config.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    
    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=5, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument('--s_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of post softmax activation')
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_s{}_calibsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.calib_size)
    else:
        auto_name = '{}_w{}_a{}_s{}_optimsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.optim_size)
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
    ckpt = torch.load(ckpt_path)
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model


def finish_training(model):
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()

    
def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    cfg.s_bit = args.s_bit if hasattr(args, 's_bit') else cfg.s_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
        
    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = 'cuda:0'
    device = torch.device(args.device)
    
    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',
    }

    seed_all(args.seed)
    
    logging.info('Building model ...')
    try:
        model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(model_zoo[args.model]))
    except:
        model = timm.create_model(model_zoo[args.model], pretrained=True)
    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()
    model.to(device)
    model.eval()
    data_path = args.dataset
    g = mydatasets.ViTImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})
    
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    logging.info('Wraping quantiztion modules (reparam: {}) ...'.format(reparam))
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device)
    model.eval()
    
    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
            if args.test_calibrate_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
        else:
            logging.info("{} - start calibration".format(get_cur_time()))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - calibration finished.".format(get_cur_time()))
            if not args.optimize:
                finish_training(model)
            save_model(model, args, cfg, mode='calibrate')
            logging.info('Validating after calibration ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    
    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start block reconstruction".format(get_cur_time()))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader)
        block_reconstructor.reconstruct_model(quant_act=cfg.train_act, keep_gpu=cfg.keep_gpu)
        finish_training(model)
        logging.info("{} - block reconstruction finished.".format(get_cur_time()))
        save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    logging.info("{} - finished the process.".format(get_cur_time()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    