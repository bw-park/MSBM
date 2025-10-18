from __future__ import absolute_import, division, print_function, unicode_literals
import colored_traceback.always
import sys
from ipdb import set_trace as debug
import pathlib
import logging
import torch
from runner import Runner
import util
import options
# from git_utils import *
import wandb

print(util.yellow("======================================================="))
print(util.yellow("Multi Marginal Schrödinger Bridge Matching"))
print(util.yellow("======================================================="))
print(util.magenta("setting configurations..."))
opt = options.set()
def main(opt):
    # run_dir = pathlib.Path("results") / opt.dir
    # setup_logger(run_dir)
    # log_git_info(run_dir)
    # log = logging.getLogger(__name__)  
    # log.info("Command used:\n{}".format(" ".join(sys.argv)))
    run = Runner(opt)
    wandb.init(project="msbm", config=opt, save_code=True, mode="online")
    # ====== Training functions ======
    if opt.mode=='train':
        run.msbm_alternate_train(opt)
    elif opt.load is not None:
        #Test this function
        run.evaluate(opt, 999, 'forward')
    else:
        raise RuntimeError()

main(opt)
