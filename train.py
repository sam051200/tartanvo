import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm, trange

from torch.utils.data import DataLoader

from Datasets.tartanVODatset import TartanVODataset
from Datasets.utils import (Compose, CropCenter, DownscaleFlow, SampleNormalize, ToTensor,
                            dataset_intrinsics, load_kiiti_intrinsics)
from TartanVO import TartanVO


def get_args():
    parser = argparse.ArgumentParser(description='HRL')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=None,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model (default: "")')
    parser.add_argument('--euroc', action='store_true', default=False,
                        help='euroc test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--flow-dir', default='',
                        help='test trajectory folder where the optical flow are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    tartanvo = TartanVO(args.model_name)
    model = tartanvo.vonet

    # load trajectory data from a folder
    datastr = 'tartanair'
    if args.kitti:
        datastr = 'kitti'
    elif args.euroc:
        datastr = 'euroc'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr)
    if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(
            args.kitti_intrinsics_file)

    transform = Compose([
        CropCenter((args.image_height, args.image_width)),
        DownscaleFlow(),
        SampleNormalize(flow_norm=20),  # pose normalize args is omitted
        ToTensor()
    ])

    train_set = TartanVODataset(args.test_dir,  posefile=args.pose_file, flowDir=args.flow_dir, transform=transform,
                                focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    num_workers = os.cpu_count() if args.worker_num is None else args.worker_num
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=num_workers)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    epoch = 10
    weight_lambda = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()
    for ep in trange(epoch, desc="#Epoch"):
        for sample in tqdm(train_loader, desc="#Batch"):
            img0 = sample['img1'].to(device)
            img1 = sample['img2'].to(device)
            intrinsic = sample['intrinsic'].to(device)
            flow_gt = sample["flow"] # N x C x H x W
            pose_gt = sample["motion"] # N x 6

            # Forward
            flow, pose = model([img0, img1, intrinsic])
            # loss = weight_lambda * criterion(flow, flow_gt) + 
            #

