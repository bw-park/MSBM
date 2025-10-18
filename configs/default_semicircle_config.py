import ml_collections
# python main.py  --dir semicircle --problem-name semicircle  --log-tb  --ckpt-freq 10 
def get_semicircle_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.seed         = 42
  config.T            = 1.0
  config.interval     = 100
  config.t0           = 0
  config.problem_name = 'semicircle'
  config.dir          = 'reproduce/semicircle'
  config.num_itr      = 200 # 1000
  config.num_epoch    = 1
  config.num_stage    = 100 # 20
  # config.eval_itr     = 200
  config.forward_net  = 'toy'
  config.backward_net = 'toy'
  config.use_arange_t = True
  config.time_scale      = 1 # 4/5
  config.train_bs_x   = 256
  config.use_amp      = True
  config.var          = 0.2 # 0.5
  config.reg          = 0.5
  # sampling
  config.samp_bs      = 2000
  config.sde_type     = 'simple'
  config.ckpt_freq    = 5
  
  # optimization
  config.weight_decay = 0
  config.optimizer    = 'AdamW'
  config.lr           = 1e-4
  config.lr_gamma   = 0.999

  model_configs=None
  return config, model_configs