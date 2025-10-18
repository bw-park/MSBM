import ml_collections

# python main.py --problem-name petal --log-tb  --ckpt-freq 10  --dir petal/exact-same

def get_hesc_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 0
  config.T            = 1.0
  config.interval     = 30
  config.t0           = 0
  config.problem_name = 'hesc'
  config.dir          = 'reproduce/hesc'
  config.num_itr      = 1000 
  config.num_epoch    = 1
  config.num_stage    = 100 
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.time_scale      = 1
  config.train_bs_x   = 256
  config.num_ResNet   = 1
  config.use_amp      = True 
  config.var          = 0.1 
  # sampling
  config.samp_bs      = 200
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 1e-3
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs

3