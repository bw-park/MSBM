import ml_collections

# main.py --problem-name RNAsc --log-tb --ckpt-freq 5  --dir RNA                                                                

def get_eb5_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 0
  config.T            = 1.0
  config.interval     = 100 #400
  config.t0           = 1e-2
  config.problem_name = 'eb5'
  config.dir          = 'reproduce/eb5'
  config.num_itr      = 1000 # 20000
  config.num_epoch    = 1
  config.num_stage    = 50
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.time_scale    = 1.
  config.train_bs_x   = 256
  config.use_amp      = True
  config.var          = 0.5
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
  config.lr           = 1e-4 
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs

