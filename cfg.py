import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=1, help='batch size')
    parser.add_argument('-lr', type=float, default=0.00008, help='learning rate')
    parser.add_argument('-ms', type=str, default='b', help='model size')
    parser.add_argument('-dataset', type=str, default='fmb', help='dataset type')
    parser.add_argument('-label_path', type=str, default=None, help='label_path')
    parser.add_argument('-ckpt', type=str, default=None, help='sam2 checkpoint path')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-gpu_device', default=0 ,type=int,help='GPU id')
    parser.add_argument('-modal_type', type=int, default=2 , help='modal type')
    parser.add_argument('-ddp', type=bool, default = False, help='if use multi gpu')
    parser.add_argument(
    '-data_path',
    type=str,
    default='/root/autodl-tmp/segment-anything-2/data/FMB',
    help='The path of segmentation data')
    opt = parser.parse_args()

    return opt
