from posixpath import split
import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch
import time
import util
import loss
from ipdb import set_trace as debug
from einops import rearrange, repeat

import torch.nn as nn


def _assert_increasing(name, ts):
    assert (ts[1:] >= ts[:-1]).all(), '{} must be monotonically increasing'.format(name)

def get_ts(opt,ts, rollout, dist_idx):
    init_idx,term_idx       =  rollout
    assert not(opt.LOO==init_idx or opt.LOO==term_idx), 'LOO cannot be the initial or terminal distribution'
    init_t_idx, term_t_idx  = dist_idx[init_idx], dist_idx[term_idx]
    _ts                     = ts[init_t_idx:term_t_idx+1]
    return _ts

def build(opt, x_dists):
    print(util.magenta("build base sde..."))
    return SimpleSDE(opt, x_dists)


class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, x_dists):
        self.opt        = opt
        # self.dt         = opt.T/opt.interval
        self.dists      = x_dists
        # self.ts         = torch.linspace(opt.t0, opt.T, opt.interval)
        self.dist_idx   = np.linspace(0, opt.interval-1, len(self.dists)).astype(int)
        # self.dist_ts    = self.ts[self.dist_idx]
        self.bdy_xv     = {'forward':None,'backward':None}
        self.next_reuse_bdy_xv  = False
        
    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign
    
    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        dw = torch.randn_like(x)*np.sqrt(dt)
        return dw

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(  t)
        dt = self.dt if dt is None else dt
        f = 0. if f is None else f
        dw = self.dw(x,dt) if dw is None else dw
        z = torch.cat([torch.zeros_like(z,),z],dim=-1)
        return x + (f + g*z)*dt + g*dw

    def propagate_x0_trick(self, x, policy, direction):
        """ propagate x0 by a tiny step """
        t0  = torch.Tensor([0])
        dt0 = self.opt.t0 - 0
        assert dt0 > 0
        z0  = policy(x,t0)
        return self.propagate(t0, x, z0, direction, dt=dt0)

    def sample_bridge(self, x_0, x_1, t):
        t = t[..., None]
        t0, ts, tT = t[:, 0], t[:, 1], t[:, 2]
        mean_t = ((tT - ts) / (tT - t0)) * x_0 + ((ts - t0) / (tT - t0)) * x_1
        var_t = self.opt.var**2 * (ts - t0) * (tT - ts) / (tT - t0)
        z_t = torch.randn_like(x_0)
        x_t = mean_t + torch.sqrt(var_t) * z_t
        return x_t
    
    def sample_target(self, x_0, x_1, t):
        t = t[..., None]
        t0, ts, tT = t[:, 0], t[:, 1], t[:, 2]
        mean_t = x_1 - x_0
        var_t = self.opt.var**2 * (ts - t0) / (tT - ts)
        z_t = torch.randn_like(x_0)
        x_t = mean_t - torch.sqrt(var_t) * z_t
        return x_t
    
    def sample_x(self, x_dist, test=False):
        if test and self.opt.problem_name=='RNAsc':
            xs = torch.Tensor(x_dist.test_sample) #test set is way much smaller than training. We repeat it to have batch size which will not add any information for the fairness of testing. repeat first dimension of xs to be N
            xs = xs.repeat(int(self.opt.samp_bs/xs.shape[0])+1,1)
            xs = xs[0:self.opt.samp_bs,...]
        else:
            xs = x_dist.sample()
        xs = xs*self.opt.data_scale #Constant scaling for the data, default is 1.
        return xs

    def t_expand(self, opt, n_dist, t, ts):
        if n_dist == 1:
            t_ = t
        else:
            t_ = [ts[(opt.interval // n_dist) * (i+1)-1] for i in range(n_dist)]
            t_ = torch.stack(t_)
            t_ = repeat(t_, "n ->n b", b=opt.samp_bs)
            t_ = rearrange(t_,  "n b -> (n b)")
        return t_
    
    def sample_traj(self, ts, policy, init_samples, init_times, save_traj=True):

        # first we need to know whether we're doing forward or backward sampling
        direction = policy.direction
        assert direction in ['forward','backward']

        _assert_increasing('ts', ts)
        
        n_dist = init_samples.shape[1]
        x = rearrange(init_samples, 'b n d -> (n b) d', n=n_dist)
        t0s = rearrange(init_times, 'n b -> (n b) 1', n=n_dist)

        
        xs = torch.empty((x.shape[0], len(ts) + 1, *x.shape[1:])) if save_traj else None
        xs[:, 0, ...] = x
        
        # don't use tqdm for fbsde since it'll resample every itr
        _ts = tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
        for idx, t in enumerate(_ts):

            t_ = t0s + t
            z = policy(x, t_)
            dw = self.dw(x)

            if save_traj:
                xs[:,idx+1,...]=x

            x = x + z * self.dt + torch.Tensor([self.var])*dw
            
        # if direction == 'forward':
        #     xs[:, -1, ...] = x
        # else: 
        #     xs[:, -1, ...] = x
        return xs, x


class SimpleSDE(BaseSDE):
    def __init__(self, opt, x_dists, var=3.0):
        super(SimpleSDE, self).__init__(opt, x_dists)
        self.opt = opt
        self.var = opt.var
    def _f(self, x, t):
        return x

    def _g(self, t):
        return torch.Tensor([self.var])
    
    
