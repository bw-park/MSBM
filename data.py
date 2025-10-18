import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.preprocessing import StandardScaler
from prefetch_generator import BackgroundGenerator
from sklearn.model_selection import train_test_split
import util
from ipdb import set_trace as debug
from einops import rearrange
import scanpy as sc

def get_data_dim(opt, problem_name):
    return {
        'gmm':          [2],
        'semicircle':   [2],
        'checkerboard': [2],
        'RNAsc':        [opt.RNA_dim],
        'petal':        [2],
        'hesc':        [5],
        'RNA5dim' : [5],
        'afhq' : [3, 64, 64],
        'mnist' : [1, 32, 32],
        'eb5' : [5],
        'cite5' : [5],
        'multi5' : [5],
        'cite100' : [100],
        'multi100' : [100]
    }.get(problem_name)

class Sampler:
    def __init__(self, distribution, batch_size, device):
        self.distribution = distribution
        self.batch_size = batch_size
        self.device = device

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, batch):
        if batch is None:
            batch = self.batch_size
        return self.distribution.sample([batch]).to(self.device)

def build(opt):
    print(util.magenta("build problem..."))

    opt.data_dim = get_data_dim(opt, opt.problem_name)
    
    distribution_builder = {
        'gmm':          gmm_builder,
        'semicircle':   SemiCircle_builder,
        'RNAsc':        RNAsc_builder,
        'navi':         navi_builder,
        'petal':        Petal_builder,
        'hesc':         hesc_builder,
        "RNA5dim":      RNA5dim_builder,
        'eb5':          EB_builder,
        'cite5':         CITE_builder,
        'multi5':        MULTI_builder,
        'cite100':         CITE_builder,
        'multi100':        MULTI_builder,
    }.get(opt.problem_name)

    dists           = distribution_builder(opt)
    if opt.problem_name == "RNA5dim":
        opt.num_dist    = len(dists[0]) 
    else:
        opt.num_dist    = len(dists)
    
    return dists

def gmm_builder(opt):
    assert opt.problem_name == 'gmm'
    # ----- pT -----
    rad=4
    var=0.1
    dists           = [
                        UniGaussian(opt,[-0,0],var= var),
                        MixMultiVariateNormal(opt.samp_bs,num=4,radius=rad,var=var),
                        MixMultiVariateNormal(opt.samp_bs,num=8,radius=rad,var=var),
                        UniGaussian(opt,[0,0],var= var),
                    ]
    
    return dists

def navi_builder(opt):
    assert opt.problem_name == 'navi'
    # ----- pT -----
    dists           = [
                        UniGaussian(opt,[-3,0],var=0.1),
                        UniGaussian(opt,[-2,0],  var=0.1),
                        UniGaussian(opt,[0,2],  var=0.1),
                        UniGaussian(opt,[2,0],  var=0.1),
                    ]
    
    return dists
    
def gmm_builder(opt):
    assert opt.problem_name == 'gmm'
    # ----- pT -----
    rad=4
    var=0.15
    dists           = [
                        UniGaussian(opt,[0,0],var= var),
                        MixMultiVariateNormal(opt.samp_bs,num=4,radius=rad,var=var),
                        MixMultiVariateNormal(opt.samp_bs,num=8,radius=rad,var=var),
                        UniGaussian(opt,[0,0],var= var),
                    ]
    
    return dists


def threemode(opt):
    # ----- pT -----
    rad=2
    var=0.1
    dists           = [
                        UniGaussian(opt,[-0,0],var= torch.Tensor([[1,0],[0,0.1]])),
                        MixMultiVariateNormal(opt.samp_bs,num=2,radius=rad,var=var,bias=torch.Tensor([0,3])),
                    ]
    
    return dists

def navi_builder(opt):
    assert opt.problem_name == 'navi'
    # ----- pT -----
    dists           = [
                        UniGaussian(opt,[-3,0],var=0.1),
                        UniGaussian(opt,[-2,0],  var=0.1),
                        UniGaussian(opt,[0,2],  var=0.1),
                        UniGaussian(opt,[2,0],  var=0.1),
                    ]
    
    return dists

def SemiCircle_builder(opt):
    assert opt.problem_name == 'semicircle'
    num     = 4
    # arc     = np.pi/(num-1)
    # vars    = [0.2*(ii+1) for ii in range(num)]
    # xs      = [np.cos(arc*idx)*5 for idx in range(num)]
    # ys      = [np.sin(arc*idx)*5 for idx in range(num)]
    # dists   = [UniGaussian(opt,[x,y],var = var) for x,y,var in zip (xs,ys, vars)]
    vars    = [0.1 for ii in range(num)]
    dists   = [UniGaussian(opt,means,var = var) for means,var in zip ([[-2,0],[-2,4],[2,4],[2,0]], vars)]
    return dists


def Petal_builder(opt):
    df      = make_diamonds(4000, 0.25, 5)
    df      = np.array(df)
    df[:,0] = df[:,0]-1
    df      = df.astype('float32')
    timestamps  = df[:,0]

    tokens      = [0,1,2,3,4]

    datasets    = []
    for token in tokens:
        data= df[np.where(token == timestamps)][:,1:]
        data= data*5

        datasets.append(data)
    dists       = [DataSampler(dataset, opt.samp_bs, opt.device) for dataset in datasets]
    return dists


def hesc_builder(opt):
    Xs_val = np.load("./data/hESC_data.npy", allow_pickle = True)
    # Xs_val = torch.FloatTensor(Xs_val)
    # Xs_val = rearrange(Xs_val, 't b d -> b t d')

    n_snap = len(Xs_val)
    tokens = torch.linspace(0, n_snap-1, n_snap).int()
    tokens = [2*i for i in range(int(n_snap/2)+1)]
    datasets    = []
    for token in tokens:
        data= Xs_val[token]
        datasets.append(data)
        
    dists       = [DataSampler(dataset, opt.samp_bs, opt.device, ratio=0) for dataset in datasets]
    return dists, [torch.tensor(data) for data in Xs_val]


def RNAsc_builder(opt):
    assert opt.problem_name == 'RNAsc'
    _dict       = np.load('data/RNAsc/ProcessedData/eb_velocity_v5.npz')
    datas        = _dict['pcs']
    scaler      = StandardScaler() #Same as TrajectoryNet implementation. see Ln332 in https://github.com/KrishnaswamyLab/TrajectoryNet/blob/master/TrajectoryNet/dataset.py.
    scaler.fit(datas)
    datas       = scaler.transform(datas)
    datas       = datas[:,0:opt.RNA_dim]
    datas       = datas.astype('float32')
    timestamps  = _dict['sample_labels']
    tokens       = np.arange(0,opt.num_marg)
    datasets    = [datas[np.where(token == timestamps)] for token in tokens]
    dists       = [DataSampler(dataset, opt.samp_bs, opt.device) for dataset in datasets]
    return dists


def RNA5dim_dataset(opt, path, is_train=False, scaler=None):
    datas, ts = [], []
    npzfile = np.load(path)
    datas.append(npzfile['X'])
    ts.append(npzfile['ts'])
    
    datas = np.concatenate(datas, axis=0)
    ts = np.concatenate(ts, axis=0)    
    
    if is_train:
        scaler = StandardScaler()
        scaler.fit(datas)
    else:
        scaler = scaler
        
    datas = scaler.transform(datas)
    datas = datas[:, 0:opt.data_dim[0]]
    datas = datas.astype('float32')
    
    tokens = np.arange(0, opt.num_marg)    
    datasets = [datas[ts == token] for token in tokens]
    
    if is_train:
        return datasets, scaler
    else:
        return datasets
    
def RNA5dim_builder(opt):
    assert opt.problem_name == 'RNA5dim'
    
    train_datasets, scaler = RNA5dim_dataset(opt, "data/RNA5dim/train_rna.npz", is_train=True)
    val_datasets = RNA5dim_dataset(opt, "data/RNA5dim/val_rna.npz", scaler=scaler)
    test_datasets = RNA5dim_dataset(opt, "data/RNA5dim/test_rna.npz", scaler=scaler)

    train_dists = [DataSampler(dataset, opt.samp_bs, opt.device, ratio=0) for dataset in train_datasets]
    val_datasets = [torch.Tensor(dataset) for dataset in val_datasets]
    test_datasets = [torch.Tensor(dataset) for dataset in test_datasets]
    
    return train_dists, val_datasets, test_datasets

    
def MULTI_builder(opt):
    print("building multi dataset...")
    embed_name = 'X_pca'
    label_name = 'day'
    max_dim = opt.data_dim[0]

    # Set your own data path for "op_train_multi_targets_0.h5ad"
    adata = sc.read_h5ad("...")
    labels = adata.obs[label_name].astype("category")
    unique_labels = labels.cat.categories.to_numpy()
    labels = labels.to_numpy()
    data = adata.obsm[embed_name][:, :max_dim]

    if opt.problem_name == "multi5":
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    ds_tensor = torch.tensor(data, dtype=torch.float32)
    label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
    frame_indices = {
        label_to_numeric[label]: (labels == label).nonzero()[0]
        for label in unique_labels
    }
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for label, indices in frame_indices.items():
        frame_data = ds_tensor[indices]
        split_index = int(len(frame_data) * 0.9)

        if len(frame_data) - split_index < opt.samp_bs:
            split_index = len(frame_data) - opt.samp_bs
        shuffled_indices = torch.randperm(len(frame_data))
        frame_data = frame_data[shuffled_indices]
        train_data = frame_data[:split_index]
        val_data = frame_data[split_index:]

        train_datasets.append(DataSampler(train_data, opt.samp_bs, device=opt.device,ratio=0))
        val_datasets.append(DataSampler(val_data, opt.samp_bs, device=opt.device,ratio=0))
        test_datasets.append(frame_data)
        
    return train_datasets, val_datasets, test_datasets

def CITE_builder(opt):
    print("building cite dataset...")
    embed_name = 'X_pca'
    label_name = 'day'
    max_dim = opt.data_dim[0]

    # Set your own data path for "op_cite_inputs_0.h5ad"
    adata = sc.read_h5ad("...")
    labels = adata.obs[label_name].astype("category")
    unique_labels = labels.cat.categories.to_numpy()
    labels = labels.to_numpy()
    data = adata.obsm[embed_name][:, :max_dim]

    if opt.problem_name == "cite5":
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    ds_tensor = torch.tensor(data, dtype=torch.float32)
    label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
    frame_indices = {
        label_to_numeric[label]: (labels == label).nonzero()[0]
        for label in unique_labels
    }
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for label, indices in frame_indices.items():
        frame_data = ds_tensor[indices]
        split_index = int(len(frame_data) * 0.9)

        if len(frame_data) - split_index < opt.samp_bs:
            split_index = len(frame_data) - opt.samp_bs
        shuffled_indices = torch.randperm(len(frame_data))
        frame_data = frame_data[shuffled_indices]
        train_data = frame_data[:split_index]
        val_data = frame_data[split_index:]

        train_datasets.append(DataSampler(train_data, opt.samp_bs, device=opt.device,ratio=0))
        val_datasets.append(DataSampler(val_data, opt.samp_bs, device=opt.device,ratio=0))
        test_datasets.append(frame_data)
        
    return train_datasets, val_datasets, test_datasets


def EB_builder(opt):
    print("building eb dataset...")
    embed_name = 'pcs'
    label_name = 'sample_labels'
    max_dim = opt.data_dim[0]

    # Set your own data path for "eb_velocity_v5.npz"
    data_dict = np.load("...", allow_pickle=True)
    data = data_dict[embed_name][:, :max_dim]
    labels = data_dict[label_name]
    unique_labels = np.unique(labels)
    
    if opt.problem_name == "eb5":
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    ds_tensor = torch.tensor(data, dtype=torch.float32)
    label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
    frame_indices = {
        label_to_numeric[label]: (labels == label).nonzero()[0]
        for label in unique_labels
    }
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for label, indices in frame_indices.items():
        frame_data = ds_tensor[indices]
        split_index = int(len(frame_data) * 0.9)

        if len(frame_data) - split_index < opt.samp_bs:
            split_index = len(frame_data) - opt.samp_bs
        shuffled_indices = torch.randperm(len(frame_data))
        frame_data = frame_data[shuffled_indices]
        train_data = frame_data[:split_index]
        val_data = frame_data[split_index:]

        train_datasets.append(DataSampler(train_data, opt.samp_bs, device=opt.device,ratio=0))
        val_datasets.append(DataSampler(val_data, opt.samp_bs, device=opt.device,ratio=0))
        test_datasets.append(frame_data)
        
    return train_datasets, val_datasets, test_datasets


class Normalize(object):
    def __call__(self, img):
        return img * 2 - 1

    def __repr__(self):
        return self.__class__.__name__

def test_loader(data, batch_dim):
    return DataLoader(
        dataset=data,
        batch_size=batch_dim,
        num_workers=2,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def setup_loader(dataset, batch_size, device):
    g=torch.Generator(device=device)
    sampler=torch.utils.data.RandomSampler(dataset,replacement=True,num_samples=batch_size,generator=torch.Generator(device=device))
    train_loader = DataLoaderX(
                                dataset, 
                                batch_size=batch_size,
                                # shuffle=True,
                                num_workers=0,
                                drop_last=True,
                                sampler=sampler,
                                generator=g,)
    # print("number of samples: {}".format(len(dataset)))

    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py#L52-L53
    # https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/train_util.py#L166
    while True:
        yield from train_loader


class DataSampler: # a dump data sampler
    def __init__(self, dataset, batch_size, device,ratio=0.15, mnist=False):
        self.num_sample = len(dataset)
        #Vanilla split dataste
        if ratio > 0:
            train_idx, val_idx  = train_test_split(list(range(len(dataset))), test_size=ratio)
            #Training data
            self.dataloader = setup_loader(dataset[train_idx, ...], batch_size, device)
            #whole dataset for evaluate as ground truth
            self.ground_truth = dataset
            # test sample
            self.test_sample = dataset[val_idx,...]
        else:
            self.dataloader = setup_loader(dataset, batch_size, device)

        self.batch_size = batch_size
        self.device = device
        self.mnist = mnist

    def sample(self):
        if self.mnist:
            data = next(self.dataloader)[0]
        else:
            data = next(self.dataloader)
        return data.to(self.device)


################################
#######Utility Functions########
################################
class UniGaussian:
    def __init__(self, opt, mean, var=1.):

        # build mu's and sigma's
        self.batch_size = opt.samp_bs
        var             = var*torch.eye(opt.data_dim[-1]) if isinstance(var,float) else var
        self.dist       = td.MultivariateNormal(torch.Tensor(mean), var)

    def sample(self):
        samples= self.dist.sample([self.batch_size])
        return samples



class MixMultiVariateNormal:
    def __init__(self, batch_size, radius=6, num=4, sigmas=None, var=1, mean=None,bias=0):


        self.bias=bias
        if mean is not None:
            mus = mean
        else: 
            # build mu's and sigma's
            arc = 2*np.pi/num
            xs = [np.cos(arc*idx)*radius for idx in range(num)]
            ys = [np.sin(arc*idx)*radius for idx in range(num)]
            mus = [torch.Tensor([x,y]) for x,y in zip(xs,ys)]

        dim = len(mus[0])
        sigmas = [var*torch.eye(dim) for _ in range(num)] if sigmas is None else sigmas

        if batch_size%num!=0:
            raise ValueError('batch size must be devided by number of gaussian')
        self.num = num
        self.batch_size = batch_size
        self.dists=[
            td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]

    def log_prob(self,x):
        # assume equally-weighted
        densities=[torch.exp(dist.log_prob(x)) for dist in self.dists]
        return torch.log(sum(densities)/len(self.dists))
    def sample(self):
        ind_sample = self.batch_size/self.num
        samples=[dist.sample([int(ind_sample)]) for dist in self.dists]
        samples=torch.cat(samples,dim=0)
        samples=samples+self.bias
        return samples

def construct_diamond(
    points_per_petal:int=200,
    petal_width:float=0.25,
    direction:str='y'
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        direction (str): Defaults to 'y'. Options `'y'` or `'x'`. Whether to make vertical
            or horizontal diamonds.
    Returns:
    ---------
        points (numpy.ndarray): the 2d array of points. 
    '''
    n_side  = int(points_per_petal/2)
    axis_1  = np.concatenate((
                np.linspace(0, petal_width, int(n_side/2)), 
                np.linspace(petal_width, 0, int(n_side/2))
            ))
    axis_2  = np.linspace(0, 1, n_side)
    axes    = (axis_1, axis_2) if direction == 'y' else (axis_2, axis_1)
    points  = np.vstack(axes).T
    points  = np.vstack((points, -1*points))
    points  = np.vstack((points, np.vstack((points[:, 0], -1*points[:, 1])).T))
    return points

def make_diamonds(
    points_per_petal:   int=200,
    petal_width:        float=0.25,
    colors:             int=5,
    scale_factor:       float=30,
    use_gaussian:       bool=True   
):
    '''
    Arguments:
    ----------
        points_per_petal (int). Defaults to `200`. Number of points per petal.
        petal_width (float): Defaults to `0.25`. How narrow the diamonds are.
        colors (int): Defaults to `5`. The number of timesteps (colors) to produce.
        scale_factor (float): Defaults to `30`. How much to scale the noise by 
            (larger values make samller noise).
        use_gaussian (bool): Defaults to `True`. Whether to use random or gaussian noise.
    Returns:
    ---------
        df (pandas.DataFrame): DataFrame with columns `samples`, `x`, `y`, where `samples`
            are the time index (corresponds to colors) 
    '''    
    upper   = construct_diamond(points_per_petal, petal_width, 'y')
    lower   = construct_diamond(points_per_petal, petal_width, 'x')
    data    = np.vstack((upper, lower)) 
    
    noise_fn    = np.random.randn if use_gaussian else np.random.rand
    noise       = noise_fn(*data.shape) / scale_factor
    data        = data + noise
    df          = pd.DataFrame(data, columns=['d1', 'd2'])
    
    c_values        = np.linspace(colors, 1, colors)
    c_thresholds    = np.linspace(1, 0+1/(colors+1), colors)
    
    df.insert(0, 'samples', colors)
    df['samples'] = colors 
    for value, threshold in zip(c_values, c_thresholds):
        index = ((np.abs(df.d1) <= threshold) & (np.abs(df.d2) <= threshold))
        df.loc[index, 'samples'] = value
    df.set_index('samples')
    return df





## From NLSB

class TrajectoryInferenceDataset(Dataset):
    def __init__(self):
        self.has_velocity = False
        pass

    def get_subset_index(self, t, n=None):
        idxs = np.arange(self.ncells)[self.labels == t]
        if not n is None:
            idxs = np.random.choice(idxs, size=n)
        return  idxs

    def get_data(self, index):
        data = dict(X=self.X[index], t=self.labels[index])
        if self.has_velocity:
            data['V'] = self.V[index]
        return data

    def get_label_set(self):
        return self.t_set

    @property
    def T0(self):
        return self.t_0

    def __len__(self):
        return self.ncells

    def __getitem__(self, index):
        x, t = self.X[index], self.labels[index]
        data = dict(x=x, t=t)
        if self.has_velocity:
            data['v'] = self.V[index]
        return data
    

class scRNASeq(TrajectoryInferenceDataset):
    def __init__(self, data_path_list, dim,  use_v=True, LMT=-1, scaler=1):
        super().__init__()
        self.dim = dim
        self.data_path_list = data_path_list
        self.t_0, self.t_T = 0.0, 4.0
        self.has_velocity = use_v

        X, ts, V = [], [], []
        for data_path in data_path_list:
            npzfile = np.load(data_path)
            X.append(npzfile['X'])
            ts.append(npzfile['ts'])
            if use_v:
                V.append(npzfile['v'])

        X = np.concatenate(X, axis=0)
        ts = np.concatenate(ts, axis=0)
        if self.has_velocity:
            V = np.concatenate(V, axis=0)
        t_set = sorted(list(set(ts[ts > 0])))

        if LMT in t_set[:-1]:
            X = X[ts != LMT]
            if self.has_velocity:
                V = V[ts != LMT]
            ts = ts[ts != LMT]

        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            self.scaler = scaler
        X = self.scaler.transform(X)
        # t_set = sorted(list(set(ts)))
        # X, ts = self.reduce_subset(X, ts, t_set)

        self._full_data = dict(X=torch.from_numpy(X[:, :dim]), t=torch.from_numpy(ts))

        self.y0 = X[ts == 0, :dim]
        self.labels = torch.from_numpy(ts[ts > 0])
        self.X = torch.from_numpy(X[ts > 0, :dim])

        self.ncells = self.X.shape[0]
        self.t_set = sorted(list(set(self.labels.numpy())))

        if self.has_velocity:
            V /= self.scaler.scale_
            self._full_data['V'] = V
            self.v0 = V[ts == 0, :dim]
            self.V = torch.from_numpy(V[ts > 0, :dim])

    @property
    def full_data(self):
        return self._full_data
    
    def get_scaler(self):
        return self.scaler
    
    def scaler_params(self):
        return { 'mean' : torch.from_numpy(self.scaler.mean_[:self.dim]).float(), 'scale' : torch.from_numpy(self.scaler.scale_[:self.dim]).float() }

    def base_sample(self, batch_size=None):
        if batch_size is None:
            x = torch.from_numpy(self.y0).float()
            if self.has_velocity:
                v = torch.from_numpy(self.v0).float()
                return dict(X=x, V=v)
        else:
            idx = np.random.choice(np.arange(len(self.y0)), size=batch_size, replace=False)
            x = torch.from_numpy(self.y0[idx]).float()

            if self.has_velocity:
                v = torch.from_numpy(self.v0[idx]).float()
                return dict(X=x, V=v)

        return dict(X=x)
    
    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_samples):
        L = len(dataset)
        self.labels = np.array([ dataset[i]['t'] for i in range(L) ])
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = len(self.labels_set)
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size