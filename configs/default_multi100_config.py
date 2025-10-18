import ml_collections

# main.py --problem-name RNAsc --log-tb --ckpt-freq 5  --dir RNA                                                                

def get_multi100_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 0
  config.T            = 1.0
  config.interval     = 100
  config.t0           = 1e-2
  config.problem_name = 'multi100'
  config.dir          = 'reproduce/multi100'
  config.num_itr      = 2000    # 1000
  config.num_epoch    = 1
  config.num_stage    = 50    # 10
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.time_scale      = 1
  config.train_bs_x   = 256
  config.use_amp      = True
  config.var          = 1.5
  config.t_scale      = 3
  config.RNA_dim      = 100 
  config.LOO          = 2
  config.num_ResNet   = 1
  # sampling
  config.samp_bs      = 4000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 1e-4      # 1e-3 or 2e-4
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs





### 0.5 

# import ml_collections

# # main.py --problem-name RNAsc --log-tb --ckpt-freq 5  --dir RNA                                                                

# def get_RNA_default_configs():
#   config = ml_collections.ConfigDict()
#   # training
#   config.training = training = ml_collections.ConfigDict()
#   config.seed         = 42
#   config.T            = 1.0
#   config.interval     = 30 #400
#   config.t0           = 1e-2
#   config.problem_name = 'RNAsc'
#   config.num_itr      = 1000
#   config.num_epoch    = 1
#   config.num_stage    = 10 # 10
#   config.forward_net  = 'toy'
#   config.backward_net = 'toy'
#   config.use_arange_t = True
#   config.time_scale      = 1
#   config.train_bs_x   = 256
#   config.use_amp      = True
#   config.var          = 0.5
#   config.RNA_dim      = 100 # 100
#   config.LOO          = -1
#   # sampling
#   config.samp_bs      = 4000
#   config.sde_type     = 'simple'
#   config.ckpt_freq    = 5
#   # optimization
#   config.weight_decay = 0
#   config.optimizer    = 'AdamW'
#   config.lr           = 1e-3 
#   config.lr_gamma   = 0.999

#   model_configs=None
#   return config, model_configs