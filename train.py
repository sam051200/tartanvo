import argparse
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
# import wandb


from Datasets.tartanVODatset import TartanVODataset
from Datasets.utils import (
    Compose,
    CropCenter,
    DownscaleFlow,
    SampleNormalize,
    ToTensor,
    dataset_intrinsics,
    load_kiiti_intrinsics,
)
from TartanVO import TartanVO

ROOT = Path(os.path.dirname(os.path.realpath(__file__)))
MODEL_DIR = ROOT / "models"


def get_args():
    parser = argparse.ArgumentParser(description="HRL")

    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size (default: 1)"
    )
    parser.add_argument(
        "--worker-num",
        type=int,
        default=None,
        help="data loader worker number (default: 1)",
    )
    parser.add_argument(
        "--image-width", type=int, default=640, help="image width (default: 640)"
    )
    parser.add_argument(
        "--image-height", type=int, default=448, help="image height (default: 448)"
    )
    parser.add_argument(
        "--model-name", default="", help='name of pretrained model (default: "")'
    )
    parser.add_argument(
        "--euroc",
        action="store_true",
        default=False,
        help="euroc test (default: False)",
    )
    parser.add_argument(
        "--kitti",
        action="store_true",
        default=False,
        help="kitti test (default: False)",
    )
    parser.add_argument(
        "--kitti-intrinsics-file",
        default="",
        help="kitti intrinsics file calib.txt (default: )",
    )
    parser.add_argument(
        "--test-dir",
        default="",
        help='test trajectory folder where the RGB images are (default: "")',
    )
    parser.add_argument(
        "--flow-dir",
        default="",
        help='test trajectory folder where the optical flow are (default: "")',
    )
    parser.add_argument(
        "--pose-file",
        default="",
        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")',
    )
    parser.add_argument(
        "--save-flow",
        action="store_true",
        default=False,
        help="save optical flow (default: False)",
    )

    args = parser.parse_args()

    return args


class PoseNormLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        # epsilon will auto move to the same device when compute
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, outputs, targets):
        # (From SE2se()): motion[0:3] = translation, motion[3:6] = rotation
        T_o = outputs[:, 0:3]
        R_o = outputs[:, 3:6]
        T_t = targets[:, 0:3]
        R_t = targets[:, 3:6]
        scale_o = torch.maximum(torch.norm(outputs, p=2, dim=1, keepdim=True), 
                                self.epsilon)
        scale_t = torch.maximum(torch.norm(targets, p=2, dim=1, keepdim=True), 
                                self.epsilon)
        T_loss = torch.norm((T_o / scale_o) - (T_t / scale_t), p=2, dim=1, keepdim=True)
        R_loss = torch.norm(R_o - R_t, p=2, dim=1, keepdim=True)
        return T_loss + R_loss


if __name__ == "__main__":
    args = get_args()
    #Monitoring
    # if True:
    #     import wandb
    #     wandb.init(project='tartanvo', name="tartanvo_org", config=cfg, sync_tensorboard=True)
    # writer = SummaryWriter(cfg.name)


    # load trajectory data from a folder
    datastr = "tartanair"
    if args.kitti:
        datastr = "kitti"
    elif args.euroc:
        datastr = "euroc"
    else:
        datastr = "tartanair"
    focalx, focaly, centerx, centery = dataset_intrinsics(datastr)
    if args.kitti_intrinsics_file.endswith(".txt") and datastr == "kitti":
        focalx, focaly, centerx, centery = load_kiiti_intrinsics(
            args.kitti_intrinsics_file
        )

    transform = Compose(
        [
            CropCenter((args.image_height, args.image_width)),
            DownscaleFlow(),
            SampleNormalize(flow_norm=20),  # pose normalize args is omitted
            ToTensor(),
        ]
    )

    train_set = TartanVODataset(
        args.test_dir,
        posefile=args.pose_file,
        flowDir=args.flow_dir,
        transform=transform,
        focalx=focalx,
        focaly=focaly,
        centerx=centerx,
        centery=centery,
    )
    num_workers = os.cpu_count() if args.worker_num is None else args.worker_num
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers
    )

    epoch = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tartanvo = TartanVO(args.model_name)
    model = tartanvo.vonet

    flow_criterion = nn.MSELoss(reduction="mean")
    pose_criterion = PoseNormLoss(1e-6)
    weight_lambda = 0.1
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    writer = SummaryWriter("runs/TartanVO")

    model.train()
    best_loss = math.inf
    for ep in trange(epoch, desc="#Epoch"):
        flow_loss = pose_loss = loss = 0
        for sample in tqdm(train_loader, desc="#Batch", leave=False):
            # Data
            img0 = sample["img1"].to(device)
            img1 = sample["img2"].to(device)
            intrinsic = sample["intrinsic"].to(device)
            # Ground truth
            flow_gt = sample["flow"].to(device)  # N x C x H x W
            pose_gt = sample["motion"].to(device)  # N x 6

            # Forward
            flow, pose = model([img0, img1, intrinsic])

            # Loss
            # Pre-divide by batch size to make it numerically stable
            batch_flow_loss = flow_criterion(flow, flow_gt)  # /batche_size, scalar
            batch_pose_loss = pose_criterion(
                pose, pose_gt
            ).mean()  # /batche_size, scalar
            batch_loss = weight_lambda * batch_flow_loss + batch_pose_loss
            
            # Total loss
            # Not accurate (a little bit) if all batch size are not the same
            flow_loss += batch_flow_loss.item()
            pose_loss += batch_pose_loss.item()
            loss += batch_loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        # Log
        writer.add_scalar("Flow loss", flow_loss, ep)
        writer.add_scalar("Pose loss", flow_loss, ep)
        writer.add_scalar("Loss", loss, ep)
        print(f"Epoch {ep}: flow_loss={flow_loss}\tpose_loss={pose_loss}\tloss={loss}")
        # TODO: validation
        # Model checkpoint
        if loss < best_loss:
            torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                }, MODEL_DIR / f"tartanvo_{datetime.now():%Y%m%d_%H%M}.tar")
        
    writer.flush()
    writer.close()
    print("\nFinished!")
