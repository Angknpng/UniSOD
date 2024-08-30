import argparse

# # #Pre-train
# parser = argparse.ArgumentParser()
# parser.add_argument("--local_rank", type=int, default=-1)
# parser.add_argument('--epoch',       type=int,   default=301,   help='epoch number')
# parser.add_argument('--lr',          type=float, default=1e-5,  help='learning rate')
# parser.add_argument('--batchsize',   type=int,   default=4,    help='training batch size')
# parser.add_argument('--trainsize',   type=int,   default=384,   help='training dataset size')
# parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
# parser.add_argument('--lw',          type=float, default=0.001, help='weight')
# parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
# parser.add_argument('--decay_epoch', type=int,   default=40,    help='every n epochs decay learning rate')

# parser.add_argument('--load',        type=str,   default='/swin_base_patch4_window12_384_22k.pth',  help='train from checkpoints')
# parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')
# parser.add_argument('--rgb_label_root',      type=str, default='/datasets/SOD/RGB/Train/RGB/',           help='the training rgb images root')#/media/data/lcl_e/datasets/SOD/RGBD/Train1/RGB/
# parser.add_argument('--gt_label_root',       type=str, default='/datasets/SOD/RGB/Train/GT/',            help='the training gt images root')#/media/data/lcl_e/datasets/SOD/RGBD/Train1/GT/
# parser.add_argument('--val_rgb_root',        type=str, default='/datasets/SOD/RGB/ECSSD/RGB/',      help='the test rgb images root')
# parser.add_argument('--val_gt_root',         type=str, default='/datasets/SOD/RGB/ECSSD/GT/',       help='the test gt images root')

# parser.add_argument('--save_path',           type=str, default='./Checkpoint/Pretrained/',    help='the path to save models and logs')

# opt = parser.parse_args()

# Fine-tuning
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--epoch',       type=int,   default=301,   help='epoch number')
parser.add_argument('--lr',          type=float, default=1e-5,  help='learning rate')
parser.add_argument('--batchsize',   type=int,   default=8,    help='training batch size')
parser.add_argument('--trainsize',   type=int,   default=384,   help='training dataset size')
parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
parser.add_argument('--lw',          type=float, default=0.001, help='weight')
parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int,   default=40,    help='every n epochs decay learning rate')

parser.add_argument('--load',        type=str,   default='/RGB_Pretrained.pth',  help='train from checkpoints')
parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')
parser.add_argument('--rgb_label_root',      type=str, default='/datasets/SOD/Train_unified/RGB/',           help='the training rgb images root')#/media/data/lcl_e/datasets/SOD/RGBD/Train1/RGB/
parser.add_argument('--depth_label_root',    type=str, default='/datasets/SOD/Train_unified/T/',         help='the training depth images root')#/media/data/lcl_e/datasets/SOD/RGBD/Train1/Depth/
parser.add_argument('--gt_label_root',       type=str, default='/datasets/SOD/Train_unified/GT/',            help='the training gt images root')#/media/data/lcl_e/datasets/SOD/RGBD/Train1/GT/

parser.add_argument('--val_rgb_root',        type=str, default='/datasets/SOD/Test_unified/RGB/',      help='the test rgb images root')
parser.add_argument('--val_depth_root',      type=str, default='/datasets/SOD/Test_unified/T/',    help='the test depth images root')
parser.add_argument('--val_gt_root',         type=str, default='/datasets/SOD/Test_unified/GT/',       help='the test gt images root')

parser.add_argument('--save_path',           type=str, default='./Checkpoint/UniSOD/',    help='the path to save models and logs')

opt = parser.parse_args()