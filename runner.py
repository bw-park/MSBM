
import os, time, gc
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from metrics import MMD_loss,compute_metrics,metric_build
import policy
import sde
from loss import compute_sb_DSB_train
import data
import util

from ipdb import set_trace as debug
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio
import wandb

def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer   = optim_name(policy.parameters(), **optim_dict)
    ema         = ExponentialMovingAverage(policy.parameters(), decay=0.999)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy

class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()

        self.opt = opt
        self.start_time = time.time()
        
        if opt.problem_name == "RNA5dim":
            dists = data.build(opt)
            self.x_dists  = dists[0]
            self.val_dists = [dist for dist in dists[1]]
            self.test_dists = [dist for dist in dists[-1]]        
        elif opt.problem_name == "hesc":
            self.x_dists, self.gt = data.build(opt)    
            # opt.interval += 1
        elif opt.problem_name == "eb5" or opt.problem_name == "cite5" or opt.problem_name == "multi5" or opt.problem_name == "cite100" or opt.problem_name == "multi100":
            self.x_dists, self.val_dists, self.test_dists = data.build(opt)
        else:
            self.x_dists  = data.build(opt)

        self.num_dist = len(self.x_dists)
        self.t_dists  = torch.linspace(0, self.num_dist-1, self.num_dist) * opt.t_scale

        # self.t_gap = self.t_dists[1:] -  self.t_dists[:-1]
        # self.opt.T = 1. #self.t_dists[1] - self.t_dists[0]
        
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)
        # self.ts[0] = opt.t0 + 1e-3
        # self.ts[-1] -= 1e-3
        self.dt = self.ts[1] - self.ts[0]
        # self.t_T = opt.T - self.dt * 0.1
        # self.t_T = self.t_gap - self.dt * 0.1
        
        self.eval_ts = [torch.linspace(self.t_dists[i], self.t_dists[i+1], opt.interval) for i in range(self.num_dist - 1)]
        self.eval_ts = torch.unique(torch.hstack(self.eval_ts))
        # self.eval_ts[0] += 1e-3
        # self.eval_ts[-1] -= 1e-3
        # for visualize training data
        if opt.problem_name =='RNAsc':
            self.x_data = [dist.ground_truth for dist in self.x_dists] 
            
        self.xs_data_ = [self.x_dists[i].sample()[:, None] for i in range(self.num_dist)]
        self.xs_data = torch.hstack(self.xs_data_)
        
        if self.opt.LOO != -1:
            assert not(opt.LOO==0 or opt.LOO>=self.num_dist-1)
            indices = [i for i in range(self.num_dist) if i != self.opt.LOO]
            self.indices = torch.tensor(indices).to(opt.device)
            self.t_dists  = self.t_dists[self.indices]
            
            
        # Build metrics
        self.metrics    = metric_build(opt)
        # build dynamics, forward (z_f) and backward (z_b) policies and corresponding optimizer
        self.dyn        = sde.build(opt, self.x_dists)
        self.dyn.dt = self.dt.cpu()

        self.z_f        = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b        = policy.build(opt, self.dyn, 'backward') # q -> p

        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)


        if opt.load:
            util.restore_checkpoint(opt, self, opt.load)

    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def evaluate(self, opt, stage, direction):
        policy_impt = {
            'forward': self.z_f, # sample from forward
            'backward': self.z_b, # sample from backward
        }.get(direction)

        t0s = self.t_dists[0][None]
        t0s = repeat(t0s, "n ->n b", b=opt.samp_bs)
        
        xs = self.xs_data
        
        if opt.load is None and direction =='forward':
            test=False
            if stage == opt.num_stage or stage == opt.num_stage - 1 :
                keys = ['z_f','optimizer_f','ema_f','z_b','optimizer_b','ema_b']
                util.save_checkpoint(opt, self, keys, stage)
        else:
            test=True
            
        _, ema_impt, _      = self.get_optimizer_ema_sched(policy_impt)
        with ema_impt.average_parameters():
            policy_impt     = freeze_policy(policy_impt)

            if direction == "forward": # sample forward
                
                if test is True:
                    if self.opt.problem_name == 'RNAsc':
                        print("Test || RNAsc")
                        # We follow the setting from DMSB in this experiment:
                        # DMSB : Test set is way much smaller than training. 
                        # DMSB : We repeat it to have batch size which will not add any information for the fairness of testing. 
                        # DMSB : repeat first dimension of xs to be N
                        init_samples = torch.Tensor(self.x_dists[0].test_sample)
                        init_samples = init_samples.repeat(
                            int(self.opt.samp_bs/init_samples.shape[0]) + 1, 1)
                        init_samples = init_samples[0:self.opt.samp_bs, None, ...]
                        init_times = t0s
                        # DMSB : we use whole data which is similar to FID computation
                        target_samples = self.xs_data_ # self.x_data 
                        ts = self.eval_ts     
                    elif self.opt.problem_name == 'RNA5dim':
                        print("Test || RNA5dim")
                        init_samples = [self.test_dists[i][:, None] for i in range(self.num_dist-1)]
                        init_samples = torch.hstack(init_samples)
                        target_samples = self.test_dists                                         
                        t0s = self.t_dists[:-1]                        
                        sample_size = init_samples.shape[0]                        
                        init_times = repeat(t0s, "n ->n b", b=sample_size)                        
                        ts = self.ts
                    elif self.opt.problem_name == 'hesc':
                        init_samples = self.gt[0][:, None]
                        init_times = t0s[:, :init_samples.shape[0]]
                        ts = self.eval_ts
                        target_samples = self.gt
                    elif self.opt.problem_name == 'eb5' or self.opt.problem_name == 'cite5' or self.opt.problem_name == 'multi5' or self.opt.problem_name == 'cite100' or self.opt.problem_name == 'multi100':
                        init_samples = [self.test_dists[i][:, None] for i in range(self.num_dist-1)]
                        init_samples = torch.hstack(init_samples)
                        target_samples = self.test_dists                                         
                        t0s = self.t_dists[:-1]                        
                        sample_size = init_samples.shape[0]                        
                        init_times = repeat(t0s, "n ->n b", b=sample_size)                        
                        ts = self.ts
                    else:
                        init_samples = xs[:, :1, :]
                        init_times = t0s
                        ts = self.eval_ts     
                        target_samples = None 
                        
                else:
                    if self.opt.problem_name == 'RNA5dim':
                        init_samples = [self.test_dists[i][:, None] for i in range(self.num_dist-1)]
                        init_samples = torch.hstack(init_samples)
                        target_samples = self.test_dists                                         
                        t0s = self.t_dists[:-1]                        
                        sample_size = init_samples.shape[0]                        
                        init_times = repeat(t0s, "n ->n b", b=sample_size)                        
                        ts = self.ts
                    elif self.opt.problem_name == 'hesc':
                        init_samples = self.gt[0][:, None]
                        init_times = t0s[:, :init_samples.shape[0]]
                        ts = self.eval_ts
                        target_samples = self.gt
                    elif self.opt.problem_name == 'RNAsc':
                        init_samples = xs[:, :1, :]
                        init_times = t0s               
                        ts = self.eval_ts    
                        
                        xxx = [xyz.squeeze(1).cpu().numpy() for xyz in self.xs_data_ ]
                        target_samples = xxx # self.x_data
                    elif self.opt.problem_name == 'eb5' or self.opt.problem_name == 'cite5' or self.opt.problem_name == 'multi5' or self.opt.problem_name == 'cite100' or self.opt.problem_name == 'multi100':
                        init_samples = [self.test_dists[self.opt.LOO -1][:, None]]
                        init_samples = torch.hstack(init_samples)
                        target_samples = self.test_dists[self.opt.LOO]                                         
                        t0s = self.t_dists[self.opt.LOO-1][None]                        
                        sample_size = init_samples.shape[0]                        
                        init_times = repeat(t0s, "n ->n b", b=sample_size)                        
                        ts = self.ts
                    else:
                        init_samples = xs[:, :1, :]
                        init_times = t0s               
                        ts = self.eval_ts    
                        target_samples = None
                    
                    
                sample_xs, _ = self.dyn.sample_traj(ts, policy_impt, init_samples, init_times)

                if self.opt.problem_name == "RNAsc":
                    compute_metrics(opt, sample_xs.cpu().numpy(), target_samples, self.metrics, self, stage, "forward", test)            
                    fn = os.path.join(opt.forward_generated_data_path, 'forward_path_stage_{}'.format(stage))
                    self.save_trajectories_pdf(xs.cpu(), sample_xs.cpu(), self.eval_ts, 
                                            fn, direction=policy_impt.direction)
                elif self.opt.problem_name == "RNA5dim":
                    sample_xs = rearrange(sample_xs[:, -1], '(n b) d -> b n d', b=sample_size)
                    compute_metrics(opt, sample_xs.cpu(), target_samples, self.metrics, self, stage, "forward", test)   
                elif self.opt.problem_name == "hesc":
                    compute_metrics(opt, sample_xs.cpu().numpy(), target_samples, self.metrics, self, stage, "forward", test)  
                    fn = os.path.join(opt.forward_generated_data_path, 'forward_path_stage_{}'.format(stage))
                    self.save_trajectories_pdf(self.gt, sample_xs.cpu(), self.eval_ts, 
                                            fn, direction=policy_impt.direction)
                elif self.opt.problem_name == "eb5" or self.opt.problem_name == "cite5" or self.opt.problem_name == "multi5" or self.opt.problem_name == "cite100" or self.opt.problem_name == "multi100":
                    _, avg_metric = compute_metrics(opt, sample_xs, target_samples, self.metrics, self, stage, "forward", test)  
                    wandb.log({"avg_metric": avg_metric})
                else:
                    fn = os.path.join(opt.forward_generated_data_path, 'forward_path_stage_{}'.format(stage))
                    self.save_trajectories_pdf(xs.cpu(), sample_xs.cpu(), self.eval_ts, 
                                            fn, direction=policy_impt.direction)
                    
                                                          
    @torch.no_grad()
    def sample_msbm_coupling(self, opt, stage, direction, policy_opt, policy_impt):
        t0s, t1s = self.t_dists[:-1], self.t_dists[1:]
        
        # x0 = [self.x_dists[i].sample()[:, None] for i in range(self.num_dist-1)]
        # x1 = [self.x_dists[i+1].sample()[:, None] for i in range(self.num_dist-1)]
        
        xs = [
            self.x_dists[i].sample()[:, None]
            for i in (self.indices if self.opt.LOO != -1 else range(self.num_dist))]
        xs = torch.hstack(xs)
        
        train_t0s = repeat(t0s, "n ->n b", b=opt.samp_bs)
        train_t1s = repeat(t1s, "n ->n b", b=opt.samp_bs)
        # Direction mean the sampling direction the "forward" means t_0 -> t_1.
        # In this direction, we train the "backward" control.
        # "backward" control try to predict X_0 given X_t, where X_t sampling from DB(\mu_i, \rho_i+1)
        if stage == 1:  
            train_x0s = xs[:, 1:, :]
            train_x1s = xs[:, :-1, :]
            train_xs = None
            
            train_t0s = train_t0s.flip(dims=[0])
            train_t1s = train_t1s.flip(dims=[0])

            print('generate train data from [{}]!'.format(util.red('Independent Coupling')))

        elif stage > 1:
            _, ema_impt, _      = self.get_optimizer_ema_sched(policy_impt)
            with ema_impt.average_parameters():
                policy_impt     = freeze_policy(policy_impt)
                
                if direction == "backward":  # train forward | sample backward
                    inits = xs[:, 1:, :]
                    train_xs, _ = self.dyn.sample_traj(self.ts, policy_impt, inits, train_t0s.flip(dims=[0]))

                    train_dists = rearrange(train_xs[:, -1], '(n b) d -> b n d', b=opt.samp_bs)
                    train_x0s = train_dists          
                    train_x1s = inits                
                                            
                elif direction == "forward":  # train backward | sample forward
                    inits = xs[:, :-1, :]
                    train_xs, _  = self.dyn.sample_traj(self.ts, policy_impt, inits, train_t0s)

                    train_dists = rearrange(train_xs[:, -1], '(n b) d -> b n d', b=opt.samp_bs)
                    train_x0s = train_dists 
                    train_x1s = inits

                    train_t0s = train_t0s.flip(dims=[0])
                    train_t1s = train_t1s.flip(dims=[0])

        else: return
        train_x0s = rearrange(train_x0s, 'b n d -> (n b) d')
        if train_x1s.shape[0] == opt.samp_bs:
            train_x1s = rearrange(train_x1s, 'b n d -> (n b) d')

        train_t0s = rearrange(train_t0s,  "n b -> (n b) 1")
        if train_t1s.shape[1] == opt.samp_bs:
            train_t1s = rearrange(train_t1s,  "n b -> (n b) 1")
        
        assert train_x0s.shape[0] == train_x1s.shape[0]
        assert train_t0s.shape[0] == train_t1s.shape[0]
        assert train_x0s.shape[0] == train_t0s.shape[0]
        
        gc.collect()
        
        return train_x0s, train_x1s, train_t0s, train_t1s

    def msbm_alternate_train(self, opt):
        
        # self.save_samples_pdf(opt.problem_name)

        bridge_ep = opt.num_epoch
        # if opt.problem_name =='petal': bridge_ep = 1 #Special handle for petal. the distance between distributions are too close.
        for stage in range(1, opt.num_stage+1):
            if (stage % 2) != 0:
                self.msbm_alternate_train_stage(opt, stage, bridge_ep, 'forward')
                if stage > 1:
                    self.evaluate(opt, stage, 'forward')
            else:
                start_time = time.time()
                self.msbm_alternate_train_stage(opt, stage, bridge_ep, 'backward')
                end_time = time.time()
                print(f"Time taken for stage {stage}: {end_time - start_time} seconds")

        
    def msbm_alternate_train_stage(self, opt, stage, epoch, direction):
        policy_opt, policy_impt = {
            'forward':  [self.z_b, self.z_f], # train backward, sample from forward
            'backward': [self.z_f, self.z_b], # train forward, sample from backward
        }.get(direction)

        for ep in range(epoch):
            # prepare training data
            train_x0s, train_x1s, train_t0s, train_t1s = self.sample_msbm_coupling(
                opt, stage, direction, policy_opt, policy_impt)   
            
            # train one epoch
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            
            self.msbm_alternate_train_ep(
                opt, stage, ep, direction, 
                train_x0s, train_x1s, train_t0s, train_t1s, policy_opt
            )

    def msbm_alternate_train_ep(self, opt, stage, epoch, direction, 
                                train_x0s, train_x1s, train_t0s, train_t1s, policy):
        
        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)
        use_amp = opt.use_amp
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        for it in range(opt.num_itr):
            # -------- sample x_idx and t_idx \in [0, interval] --------
            samp_x_idx = torch.randint(opt.samp_bs * (len(self.t_dists)-1),  (opt.train_bs_x,),device='cpu')
            
            train_x0s_sample, train_x1s_sample = train_x0s[samp_x_idx], train_x1s[samp_x_idx]
            train_t0s_sample, train_t1s_sample = train_t0s[samp_x_idx], train_t1s[samp_x_idx]
            
            t_T = (train_t1s_sample - train_t0s_sample) - self.dt * 0.1

            train_t = (torch.rand(size=(opt.train_bs_x, 1)).to(opt.device) * t_T)
            
            train_t_sample = torch.cat([train_t0s_sample, train_t0s_sample + train_t, train_t1s_sample], dim=-1)
                
            train_xts_sample = self.dyn.sample_bridge(train_x0s_sample, train_x1s_sample, train_t_sample)
            train_xts_target = self.dyn.sample_target(train_x0s_sample, train_x1s_sample, train_t_sample)
            
            alpha_t = policy(train_xts_sample, train_t_sample[:, 1])
            
            loss = torch.mean((train_xts_target - alpha_t) ** 2)
            
            scaler.scale(loss).backward()
            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()
            
            self.log_sb_alternate_train(opt, it, epoch, stage, loss, optimizer, direction, opt.num_epoch)

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, optimizer, direction, num_epoch):
        time_elapsed = util.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        if (it+1)%1000==0:
            print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
                .format(
                    util.magenta("SB {} sampling".format(direction)),
                    util.cyan("{}".format(stage)),
                    opt.num_stage,
                    util.cyan("{}".format(1+ep)),
                    num_epoch,
                    util.cyan("{}".format(1+it+opt.num_itr*ep)),
                    opt.num_itr*num_epoch,
                    util.yellow("{:.2e}".format(lr)),
                    util.red("{:+.4f}".format(loss.item())),
                    util.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
            ))

    def save_trajectories_pdf(self, xs, traj, ts, file_name, sample_size=100, direction='forward'):
        
        if traj.shape[0] != self.opt.samp_bs:
            traj = rearrange(traj, "(n b) t d -> b (n t) d", n=self.num_dist-1, t=len(ts) + 1, d=self.opt.data_dim[0])
        np.save(file_name + '_path_seed_{}.npy'.format(self.opt.seed), traj)
        # traj = rearrange(traj, "(n b) t d -> b (n t) d", n=4, t=101)
        traj = traj[:sample_size]
        num_dists = traj.shape[1]
        cmap = cm.get_cmap('viridis', num_dists)
        colors = [cmap(i) for i in range(num_dists)]
        
        num_dists_ = len(self.x_dists) 
        
        if self.opt.problem_name == "RNAsc":
            xs = [torch.Tensor(self.x_data[i])[:sample_size, None] for i in range(self.num_dist)]
            xs = torch.hstack(xs).cpu()
        elif self.opt.problem_name == "hesc":
            num_dists_ = len(self.gt)
            # sample_size = xs.size(0)
            # xs = xs[:sample_size].cpu()
        else:
            xs = xs[:sample_size].cpu()

        fig, ax = plt.subplots(figsize=(6, 5))
        
        for i in range(num_dists_):
            if self.opt.problem_name == "hesc":
                xs__ = self.gt[i].cpu()
            else:
                xs__ = xs[:, i]
            ax.scatter(xs__[:, 0], xs__[:, 1], s=10, color='k', zorder=0, alpha=0.2)
            
        for i in range(num_dists):
        # for i in range(5):
            xs__ = traj[:, i]
            # ax.scatter(xs__[:, 0], xs__[:, 1], s=1, color=colors[i], alpha=0.1)
            ax.scatter(xs__[:, 0], xs__[:, 1], s=2, color=colors[i], alpha=1., zorder=1)
            # ax.plot(xs__[:, 0], xs__[:, 1], color='gray', alpha=0.5, lw=0.08)
        if self.opt.problem_name == "petal":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5
        # elif file_name == "semicircle_traj":
        elif self.opt.problem_name == "semicircle":
            x_lim_min, x_lim_max = -4, 4
            y_lim_min, y_lim_max = -1.5, 5.5
        elif self.opt.problem_name == "gmm":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5           
        elif self.opt.problem_name == "RNAsc":
            x_lim_min, x_lim_max = -4.0, 4.0
            y_lim_min, y_lim_max = -4.0, 4.0    
        elif self.opt.problem_name == 'hesc':
            # x_lim_min, x_lim_max = -1.2, 1.2
            # y_lim_min, y_lim_max = -1.2, 1.2   
            x_lim_min, x_lim_max = -2.5, 2.5
            y_lim_min, y_lim_max = -2.5, 2.5
                                                
        ax.set_xlim([x_lim_min, x_lim_max])
        ax.set_ylim([y_lim_min, y_lim_max])
        ax.grid(True)
        ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        # ----- Add a horizontal colorbar representing "time indices" -----
        # 1) Define normalization from 0 to num_dists-1
        norm = mcolors.Normalize(vmin=0, vmax=num_dists - 1)
        sm = cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # dummy array for colorbar

        # 3) Add the colorbar below the plot
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.05, pad=0.05, shrink=3.)
        ticks = [0, ((self.num_dist-1) * self.opt.interval)//2, (self.num_dist-1) * self.opt.interval + 1]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(["0", "1/2T", "T"])
        cbar.set_label("Time")

        plt.savefig(file_name, dpi=100)
        plt.close()


    def save_trajectories_gif(self, xs, traj, ts, file_name, sample_size=1000, direction='forward'):
        images = []

        if traj.shape[0] != self.opt.samp_bs:
            traj = rearrange(traj, "(n b) t d -> b (n t) d", n=self.num_dist-1, t=len(ts) + 1, d=2)


        traj = traj[:sample_size]
        num_dists = traj.shape[1]
        # cmap = cm.get_cmap('cividis', num_dists)
        # 1) define continuous normalization from 0 to num_dists-1
        norm = mcolors.Normalize(vmin=0, vmax=num_dists - 1)
        # 2) create ScalarMappable with cividis colormap
        cividis_map = cm.ScalarMappable(norm=norm, cmap='viridis')
        cividis_map.set_array([])  # needed for colorbar

        num_dists_ = len(self.x_dists)
        
        if self.opt.problem_name == "RNAsc":
            xs = [torch.Tensor(self.x_data[i])[:sample_size, None] for i in range(self.num_dist)]
            xs = torch.hstack(xs).cpu()
        else:
            xs = xs[:sample_size].cpu()

        if self.opt.problem_name == "petal":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5
        elif self.opt.problem_name == "semicircle":
            x_lim_min, x_lim_max = -4, 4
            y_lim_min, y_lim_max = -1.5, 5.5
        elif self.opt.problem_name == "gmm":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5
        elif self.opt.problem_name == "RNAsc":
            x_lim_min, x_lim_max = -4.0, 4.0
            y_lim_min, y_lim_max = -4.0, 4.0    
        elif self.opt.problem_name == 'hesc':
            x_lim_min, x_lim_max = -2.5, 2.5
            y_lim_min, y_lim_max = -2.5, 2.5 
            # x_lim_min, x_lim_max = -5.5, 5.5
            # y_lim_min, y_lim_max = -5.5, 5.5
                                           
        interval = 3
        for t in range(num_dists):
            if (t % interval) == 0:
                fig, ax = plt.subplots(figsize=(6, 5))

                # Scatter initial distributions
                for i in range(num_dists_):
                    xs__ = xs[:, i]
                    ax.scatter(xs__[:, 0], xs__[:, 1], s=10, color='k', zorder=0, alpha=0.2)

                # Scatter trajectories up to current time
                for i in range(t + 1):
                    
                    color_val = cividis_map.to_rgba(i)
                    
                    if i < t:
                        xs__ = traj[:, i]
                        ax.scatter(xs__[:, 0], xs__[:, 1], s=2, color=color_val, alpha=0.05, zorder=1)
                    if i == t:
                        xs__ = traj[:, i]
                        ax.scatter(xs__[:, 0], xs__[:, 1], s=10, 
                                   marker='o',
                                   facecolors='black',
                                   edgecolors=color_val, 
                                   linewidths=1.,
                                   alpha=1., zorder=1) 
                                               
                ax.set_xlim([x_lim_min, x_lim_max])
                ax.set_ylim([y_lim_min, y_lim_max])
                ax.grid(True)
                ax.tick_params(axis='both', labelbottom=False, labelleft=False)

                plt.title('T={:.2f}'.format((self.eval_ts[t])))

                # 3) add a horizontal color bar using same ScalarMappable
                cbar = plt.colorbar(cividis_map,
                                    ax=ax,
                                    orientation='horizontal',
                                    fraction=0.05,
                                    pad=0.05)
                cbar.set_label("Time")
                ticks = [0, ((self.num_dist-1) * self.opt.interval)//2, (self.num_dist-1) * self.opt.interval + 1]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(["0", "1/2T", "T"])
                cbar.set_label("Time")
        
                # Save frame to temporary buffer
                plt.savefig(file_name + '_temp_{}.png'.format(t), dpi=100)
                images.append(imageio.imread(file_name + '_temp_{}.png'.format(t)))
                plt.close()
        # Save as GIF
        imageio.mimsave(file_name + '.gif', images, fps=20)
        
        for t in range(num_dists):
            if (t % interval) == 0:
                os.remove(file_name + '_temp_{}.png'.format(t))


    def save_samples_pdf(self, file_name):
        num_dists = len(self.x_dists)
        xs = [self.x_dists[i].sample()[:, None] for i in range(num_dists)]
        xs_ = torch.hstack(xs).cpu()
        sample_size = 500
        xs_sample = xs_[:sample_size]
        if self.opt.problem_name == 'hesc':
            xs_ = self.gt
            num_dists = len(self.gt)#xs_.shape[1]
            sample_size = 111


        cmap = cm.get_cmap('viridis', num_dists)
        colors = [cmap(i) for i in range(num_dists)]
        
        plt.figure(figsize=(6, 5))
        for i in range(num_dists):
            if self.opt.problem_name == 'hesc':
                xs__ = self.gt[i].cpu()
            else:
                xs__ = xs_sample[:, i]
            plt.scatter(xs__[:, 0], xs__[:, 1], s=10, color=colors[i])
        if file_name == "petal":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5
        elif file_name == "semicircle":
            x_lim_min, x_lim_max = -4, 4
            y_lim_min, y_lim_max = -1.5, 5.5
        elif self.opt.problem_name == "gmm":
            x_lim_min, x_lim_max = -5.5, 5.5
            y_lim_min, y_lim_max = -5.5, 5.5   
        elif self.opt.problem_name == "RNAsc":
            x_lim_min, x_lim_max = -4.0, 4.0
            y_lim_min, y_lim_max = -4.0, 4.0    
        elif self.opt.problem_name == "RNA5dim":
            x_lim_min, x_lim_max = -4.0, 4.0
            y_lim_min, y_lim_max = -4.0, 4.0    
        elif self.opt.problem_name == 'hesc':
            x_lim_min, x_lim_max = -2.5, 2.5
            y_lim_min, y_lim_max = -2.5, 2.5
            # x_lim_min, x_lim_max = -5.5, 5.5
            # y_lim_min, y_lim_max = -5.5, 5.5                           

        plt.xlim([x_lim_min, x_lim_max])
        plt.ylim([y_lim_min, y_lim_max])
        plt.grid(True)
        plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        plt.savefig(f"{file_name}.png", dpi=1000)
        plt.close()
          