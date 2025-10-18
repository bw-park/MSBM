import numpy as np
import torch
import torch.nn.functional as F
from ipdb import set_trace as debug
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from ot.sliced import sliced_wasserstein_distance,max_sliced_wasserstein_distance
import util

import ot
from einops import rearrange

import math
from functools import partial
from typing import Optional

def metric_build(opt):
    metrics = {
        'SWD':sliced_wasserstein_distance,
        'MMD':MMD_loss(),
        'MWD':max_sliced_wasserstein_distance
    }
    return [metrics.get(key) for key in opt.metrics]


def compute_metrics(opt, pred_traj, ref_data, metrics, runner, stage, direction, test=False):
    ## From DMSB
    '''
    pred_traj: [batch_size, interval, data_dim] torch.Tensor
    ref_data: [num_dist, target_samples, data_dim], torch.Tensor, we use whole ref data which is similar to FID computation
    The reference data and prediction are all the marignals. We delete the leave one out (--LOO) marginal during the training, but we still evaluate them during here.
    '''
    if opt.problem_name == "RNA5dim":
        pred_data = rearrange(pred_traj, 'b n d -> n b d')  # [num_dist-1, samp_bs, data_dim]
        ref_data = ref_data[1:]
        ref_data = [sample.cpu() for sample in ref_data]
        sample_size = pred_data.shape[1]
        
        avg_metric  = 0
        for idx,(pred,ref) in enumerate(zip(pred_data, ref_data)):
            M = torch.cdist(ref, pred, p=2)
            a, b = ot.unif(ref.size()[0]), ot.unif(pred.size()[0])
            loss = ot.emd2(a, b, M.cpu().detach().numpy())
            avg_metric += loss
            
            print(util.green('[SB {} sampling] stage {}/{} | {} for time{} is {}'
                            .format(
                                direction, 
                                stage if test is False else 'test',
                                opt.num_stage if test is False else 'test',
                                "MDD", idx+1,loss)))

        avg_metric = avg_metric/(opt.num_dist-1)
        print('[SB {} sampling] stage {}/{} | AVERAGE {} IS {}'
            .format(
                direction, 
                stage if test is False else 'test',
                opt.num_stage if test is False else 'test',
                "MDD",avg_metric))
    elif opt.problem_name == "hesc":
        idxs = [15, 45] # For hesc, we set discretization steps to be 30. Hence, validation time will be 15, 45
        for i in range(len(idxs)):
            metric = earth_mover_distance(ref_data[2*i+1].cpu().numpy(), pred_traj[:, idxs[i]])
            
            print(util.green('[SB {} sampling] stage {}/{} | {} for time{} is {}'
                            .format(
                                direction, 
                                stage if test is False else 'test',
                                opt.num_stage if test is False else 'test',
                                'EMD', 2*i+1, metric)))
        pred_data = pred_traj[:, idxs].transpose(1, 0, 2)
        avg_metric = 0
    elif opt.problem_name == "eb5" or opt.problem_name == "cite5" or opt.problem_name == "multi5" or opt.problem_name == "cite100" or opt.problem_name == "multi100":
        pred_data = pred_traj[:, -1]
        avg_metric = wasserstein_distance(ref_data, pred_data, p=1)
        print('[SB {} sampling] stage {}/{} | AVERAGE {} IS {}'
            .format(
                direction, 
                stage if test is False else 'test',
                opt.num_stage if test is False else 'test',
                "MDD",avg_metric))    
            
    else:
        sample_size     = 1000
        n = opt.num_dist - 1
        dist_time       = np.linspace(0, n*(opt.interval-1), opt.num_dist).astype(int) #we delete a distribution when LOO during training, so num_dist is same as original marginal
        pred_idx        = np.random.choice(pred_traj.shape[0], sample_size, replace=False) #random sample from batch
        pred_data       = pred_traj[pred_idx][:,dist_time,0:opt.data_dim[0]] # [samp_bs, num_dist, data_dim] 
        pred_data       = pred_data.transpose(1,0,2)/opt.data_scale # [num_dist, samp_bs, data_dim]
        
        for metric_idx, metric in enumerate(metrics): #loop over metrics
            avg_metric  = 0
            for idx,(pred,ref) in enumerate(zip(pred_data, ref_data)):
                if idx==0:
                    continue # First marginal does not need to be evaluate. We do not generate it, just ground truth.
                if opt.metrics[metric_idx] == 'MMD': 
                    ref_idx = np.random.choice(ref.shape[0], sample_size, replace=False)
                    ref     = torch.Tensor(ref[ref_idx])
                    pred    = torch.Tensor(pred)

                loss        = metric(pred,ref)
                avg_metric += loss
                print(util.green('[SB {} sampling] stage {}/{} | {} for time{} is {}'
                                .format(
                                    direction, 
                                    stage if test is False else 'test',
                                    opt.num_stage if test is False else 'test',
                                    opt.metrics[metric_idx], idx,loss)))

            avg_metric = avg_metric/(opt.num_dist-1)
            print('[SB {} sampling] stage {}/{} | AVERAGE {} IS {}'
                .format(
                    direction, 
                    stage if test is False else 'test',
                    opt.num_stage if test is False else 'test',
                    opt.metrics[metric_idx],avg_metric))

    return pred_data, avg_metric

class MMD_loss(torch.nn.Module):
    '''
    fork from: https://github.com/ZongxianLee/MMD_Loss.Pytorch
    '''
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    
    
def earth_mover_distance(
    p,
    q,
    eigenvals=None,
    weights1=None,
    weights2=None,
    return_matrix=False,
    metric="sqeuclidean",
):
    """
    Returns the earth mover's distance between two point clouds
    Parameters
    ----------
    cloud1 : 2-D array
        First point cloud
    cloud2 : 2-D array
        Second point cloud
    Returns
    -------
    distance : float
        The distance between the two point clouds
    """
    p = p.toarray() if scipy.sparse.isspmatrix(p) else p
    q = q.toarray() if scipy.sparse.isspmatrix(q) else q
    if eigenvals is not None:
        p = p.dot(eigenvals)
        q = q.dot(eigenvals)
    if weights1 is None:
        p_weights = np.ones(len(p)) / len(p)
    else:
        weights1 = weights1.astype("float64")
        p_weights = weights1 / weights1.sum()

    if weights2 is None:
        q_weights = np.ones(len(q)) / len(q)
    else:
        weights2 = weights2.astype("float64")
        q_weights = weights2 / weights2.sum()

    pairwise_dist = np.ascontiguousarray(
        pairwise_distances(p, Y=q, metric=metric, n_jobs=-1)
    )

    result = ot.emd2(
        p_weights, q_weights, pairwise_dist, numItermax=1e7, return_matrix=return_matrix
    )
    if return_matrix:
        square_emd, log_dict = result
        return np.sqrt(square_emd), log_dict
    else:
        return np.sqrt(result)


def interpolate_with_ot(p0, p1, tmap, interp_frac, size):
    """
    Interpolate between p0 and p1 at fraction t_interpolate knowing a transport map from p0 to p1
    Parameters
    ----------
    p0 : 2-D array
        The genes of each cell in the source population
    p1 : 2-D array
        The genes of each cell in the destination population
    tmap : 2-D array
        A transport map from p0 to p1
    t_interpolate : float
        The fraction at which to interpolate
    size : int
        The number of cells in the interpolated population
    Returns
    -------
    p05 : 2-D array
        An interpolated population of 'size' cells
    """
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError(
            "Unable to interpolate. Tmap size is {}, expected {}".format(
                tmap.shape, (len(p0), len(p1))
            )
        )
    I = len(p0)
    J = len(p1)
    # Assume growth is exponential and retrieve growth rate at t_interpolate
    # If all sums are the same then this does not change anything
    # This only matters if sum is not the same for all rows
    p = tmap / np.power(tmap.sum(axis=0), 1.0 - interp_frac)
    p = p.flatten(order="C")
    p = p / p.sum()
    choices = np.random.choice(I * J, p=p, size=size)
    return np.asarray(
        [p0[i // J] * (1 - interp_frac) + p1[i % J] * interp_frac for i in choices],
        dtype=np.float64,
    )


def interpolate_per_point_with_ot(p0, p1, tmap, interp_frac):
    """
    Interpolate between p0 and p1 at fraction t_interpolate knowing a transport map from p0 to p1
    Parameters
    ----------
    p0 : 2-D array
        The genes of each cell in the source population
    p1 : 2-D array
        The genes of each cell in the destination population
    tmap : 2-D array
        A transport map from p0 to p1
    t_interpolate : float
        The fraction at which to interpolate
    Returns
    -------
    p05 : 2-D array
        An interpolated population of 'size' cells
    """
    assert len(p0) == len(p1)
    p0 = p0.toarray() if scipy.sparse.isspmatrix(p0) else p0
    p1 = p1.toarray() if scipy.sparse.isspmatrix(p1) else p1
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    tmap = np.asarray(tmap, dtype=np.float64)
    if p0.shape[1] != p1.shape[1]:
        raise ValueError("Unable to interpolate. Number of genes do not match")
    if p0.shape[0] != tmap.shape[0] or p1.shape[0] != tmap.shape[1]:
        raise ValueError(
            "Unable to interpolate. Tmap size is {}, expected {}".format(
                tmap.shape, (len(p0), len(p1))
            )
        )

    I = len(p0)
    J = len(p1)
    # Assume growth is exponential and retrieve growth rate at t_interpolate
    # If all sums are the same then this does not change anything
    # This only matters if sum is not the same for all rows
    p = tmap / (tmap.sum(axis=0) / 1.0 - interp_frac)
    # p = tmap / np.power(tmap.sum(axis=0), 1.0 - interp_frac)
    # p = p.flatten(order="C")
    p = p / p.sum(axis=0)
    choices = np.array([np.random.choice(I, p=p[i]) for i in range(I)])
    return np.asarray(
        [
            p0[i] * (1 - interp_frac) + p1[j] * interp_frac
            for i, j in enumerate(choices)
        ],
        dtype=np.float64,
    )


def wasserstein_distance(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 1,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

