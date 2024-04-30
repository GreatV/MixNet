from easydict import EasyDict

config = EasyDict()
config.means = 0.485, 0.456, 0.406
config.stds = 0.229, 0.224, 0.225
config.gpu = '1'
config.exp_name = 'Synthtext'
config.num_workers = 24
config.batch_size = 12
config.max_epoch = 200
config.start_epoch = 0
config.lr = 0.0001
config.cuda = True
config.output_dir = 'output'
config.input_size = 640
config.max_annotation = 64
config.adj_num = 4
config.num_points = 20
config.use_hard = True
config.load_memory = False
config.scale = 1
config.grad_clip = 25
config.dis_threshold = 0.3
config.cls_threshold = 0.8
config.approx_factor = 0.004
config.know = False
config.knownet = 'mixTriHRnet_cbam'
config.know_resume = './model/Totaltext_mid/TextBPN_mixTriHRnet_cbam_622.pth'


def update_config(cfg, extra_cfg):
    for k, v in vars(extra_cfg).items():
        cfg[k] = v
    cfg.place = str('cuda').replace('cuda', 'gpu') if cfg.cuda else str(
        'cpu').replace('cuda', 'gpu')


def print_config(cfg):
    print('==========Options============')
    for k, v in cfg.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
