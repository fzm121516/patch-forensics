import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--isTrain', action='store_true', default=False)
        parser.add_argument('--model', type=str, default='basic_discriminator', help='chooses which model to use')
        parser.add_argument('--which_model_netD', type=str, default='resnet18', help='selects model to use for netD')
        parser.add_argument('--fake_class_id', type=int, default=0, help='class id of fake ims')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_model', action='store_true', help='load the latest model')
        parser.add_argument('--seed', type=int, default=0, help='torch.manual_seed value')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')

        # image loading
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--real_im_path', type=str, help='path to real images')
        parser.add_argument('--fake_im_path', type=str, help='path to fake images')
        parser.add_argument('--no_serial_batches', action='store_true', help='if not specified, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help="Maximum number of samples to use in dataset")

        # checkpoint saving and naming 
        parser.add_argument('--name', type=str, default='', help='name of the experiment. it decides where to store samples and models')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--prefix', default='', type=str, help='customized prefix: opt.name = prefix + opt.name: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')


        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(opt.checkpoints_dir):
            os.makedirs(opt.checkpoints_dir)  # 创建 checkpoints_dir（及其上层目录）
            os.mkdir(expr_dir)  # 创建 expr_dir

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        # opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        #opt.classes = opt.classes.split(',')
        # opt.rz_interp = opt.rz_interp.split(',')
        # opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        # opt.jpg_method = opt.jpg_method.split(',')
        # opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        # if len(opt.jpg_qual) == 2:
        #     opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        # elif len(opt.jpg_qual) > 2:
        #     raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt



  