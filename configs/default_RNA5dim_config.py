import ml_collections

# main.py --problem-name RNAsc --log-tb --ckpt-freq 5  --dir RNA                                                                

def get_RNA5dim_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 0
  config.T            = 1.0
  config.interval     = 100 #400
  config.t0           = 0
  config.problem_name = 'RNA5dim'
  config.dir          = 'reproduce/RNA5dim'
  config.num_itr      = 50000 # 20000
  config.num_epoch    = 1
  config.num_stage    = 3
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.time_scale    = 1.
  config.train_bs_x   = 256
  config.use_amp      = True
  config.var          = 0.1
  config.RNA_dim      = 5 # 100
  config.LOO          = -1
  config.num_ResNet   = 1
  # sampling
  config.samp_bs      = 4000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 2e-4 
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs

