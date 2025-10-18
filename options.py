import numpy as np
import os
import argparse
import random
import torch

import configs
import util

from ipdb import set_trace as debug


def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name",   default='multi100', type=str)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,              help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,           help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default="reproduce/multi100",           help="directory name to save the experiments under results/")
    parser.add_argument("--log-tb",         default=True,                help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",                help="use cpu device")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=0,           help="time integral start time")
    parser.add_argument("--T",              type=float, default=4.,             help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=400,            help="number of interval")
    parser.add_argument("--forward-net",    type=str,   choices=['toy'],        help="model class of forward nonlinear drift")
    parser.add_argument("--backward-net",   type=str,   choices=['toy'],        help="model class of backward nonlinear drift")
    parser.add_argument("--sde-type",       type=str,   default='simple',       choices=['simple'])

    # --------------- SB training & sampling (corrector) ---------------
    parser.add_argument("--data-scale",     type=float, default=1,              help="scaling the original data")
    parser.add_argument("--LOO",            type=int, default=-1,               help="Leave One Out distribution index. -1 is not using this flag")
    parser.add_argument("--mode",           type=str, default='train',          help="traing or test mode" )
    parser.add_argument("--use-amp",        type=bool, default=True,                help="use half precision")
    parser.add_argument("--use-arange-t",   type=bool, default=True,                help="Using all timesteps to train when one has enough gpu")
    parser.add_argument("--train-bs-x",     type=int, default=256,                           help="training batch size in data dimension")
    parser.add_argument("--RNA-dim",        type=int, default=5,                help="PCA dimension of RNA")
    parser.add_argument("--num-ResNet",        type=int, default=2,             help="number of Resnet used for NN")
    parser.add_argument("--train-bs-t",     type=int,                           help="if use_arange_t is False, set the batch size for time dimension")
    parser.add_argument("--num-stage",      type=int, default=10,                            help=" number of stage")
    parser.add_argument("--num-epoch",      type=int, default=1,                            help=" number of training epoch in each stage")
    parser.add_argument("--num-marg",      type=int, default=5,                           help=" number of training epoch in each stage")
    parser.add_argument("--var",            type=float,default=0.5,               help="variance of Wiener Processß")
    parser.add_argument("--t-scale",        type=float,default=1,               help="time scale")
    parser.add_argument("--samp-bs",        type=int,default=2000,                       help="batch size for sampling")
    parser.add_argument("--num-itr",        type=int,default=1000,                      help="[sb train] number of training iterations (for each epoch)")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,                   help="learning rate")
    parser.add_argument("--lr-f",           type=float, default=None,     help="learning rate for forward network")
    parser.add_argument("--lr-b",           type=float, default=None,     help="learning rate for backward network")
    parser.add_argument("--lr-gamma",       type=float, default=0.999,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')

    # ---------------- evaluation ----------------
    parser.add_argument("--snapshot-freq",  type=int,   default=0,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=5,        help="checkpoint saving frequency w.r.t stages")
    parser.add_argument("--metrics",         type=str, default=['SWD','MWD','MMD'],       help="metric to test algorithm" )
    problem_name = parser.parse_args().problem_name
    default_config, model_configs = {
        'semicircle':       configs.get_semicircle_default_configs,
        'RNAsc':            configs.get_RNA_default_configs,
        'petal':            configs.get_petal_default_configs,
        'RNA5dim':          configs.get_RNA5dim_default_configs,
        'hesc':             configs.get_hesc_default_configs,
        'eb5':              configs.get_eb5_default_configs,
        'cite5':            configs.get_cite5_default_configs,
        'multi5':           configs.get_multi5_default_configs,
        'cite100':          configs.get_cite100_default_configs,
        'multi100':         configs.get_multi100_default_configs,
    }.get(problem_name)()
    parser.set_defaults(**default_config)

    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        # https://github.com/pytorch/pytorch/issues/7068
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.autograd.set_detect_anomaly(True)
    
    # ========= auto setup & path handle =========
    opt.device='cuda'
    opt.model_configs = model_configs
    if opt.lr is not None:
        opt.lr_f, opt.lr_b = opt.lr, opt.lr


    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    opt.eval_path = os.path.join('results', opt.dir)
    os.makedirs(os.path.join(opt.eval_path, 'forward'), exist_ok=True)
    os.makedirs(os.path.join(opt.eval_path, 'backward'), exist_ok=True)
    opt.ckpt_path = os.path.join(opt.eval_path, 'checkpoint')
    os.makedirs(opt.ckpt_path, exist_ok=True)
    
    if util.is_toy_dataset(opt):
        opt.forward_generated_data_path = os.path.join(
            'results', opt.dir, 'forward', 'generated_data'
        )
        opt.backward_generated_data_path = os.path.join(
            'results', opt.dir, 'backward', 'generated_data'
        )
        os.makedirs(opt.forward_generated_data_path, exist_ok=True)
        os.makedirs(opt.backward_generated_data_path, exist_ok=True)
    # ========= print options =========
    for o in vars(opt):
        print(util.green(o),":",util.yellow(getattr(opt,o)))
    print()
    return opt
