import re
from os import listdir

import cv2
import numpy as np
from torch.utils.data import Dataset

from .transformation import SEs2ses, pos_quats2SEs, pose2motion
from .utils import make_intrinsics_layer


class TartanVODataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgDir: str, posefile: str, flowDir: str,
                 transform=None,
                 focalx=320.0, focaly=320.0, centerx=320.0, centery=240.0):

        self.imgDir = imgDir
        self.posefile = posefile
        self.flowDir = flowDir
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

        # RGB (must)
        files = listdir(imgDir)
        self.rgbFiles = [f"{imgDir}/{f}"
                         for f in files if (f.endswith('.png') or f.endswith('.jpg'))]
        self.rgbFiles.sort()
        print(f'Find {len(self.rgbFiles)} image files in {imgDir}')

        # GT Pose (left) (optional, for training)
        if posefile is not None and posefile != "":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert (poselist.shape[1] == 7)  # position + quaternion

            poses = pos_quats2SEs(poselist)  # To SE3
            motions = pose2motion(poses)  # To relative motion (flatten SE3)
            self.motions = SEs2ses(motions).astype(np.float32)  # To se3
            assert (len(self.motions) == len(self.rgbFiles)) - 1
        else:
            self.motions = None

        # GT Optical Flow (optional, for training)
        # Not using mask to make Optical flow NN learn more from masked (dynamic) area
        if flowDir is not None and flowDir != "":
            files = listdir(flowDir)
            rxFlowFiles = re.compile(
                r"flow\.npy$", re.MULTILINE | re.IGNORECASE)
            self.flowFiles = [f"{flowDir}/{f}"
                              for f in files if re.search(rxFlowFiles, f)]
            self.flowFiles.sort()
            assert (len(self.flowFiles) == len(self.rgbFiles) - 1)
        else:
            self.flowFiles = None

    def __len__(self):
        return len(self.rgbFiles) - 1

    def __getitem__(self, idx):
        # RGB
        imgfile1 = self.rgbFiles[idx].strip()
        imgfile2 = self.rgbFiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        # Intrinsics
        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(
            w, h, self.focalx, self.focaly, self.centerx, self.centery)

        res = {
            'img1': img1,
            'img2': img2,
            'intrinsic': intrinsicLayer,
        }

        # Optical flow
        if self.flowFiles is not None:
            flowFile = self.flowFiles[idx].strip()
            res["flow"] = np.load(flowFile)

        # Transform before non-image type added to result
        if self.transform:
            res = self.transform(res)

        if self.motions is not None:
            res['motion'] = self.motions[idx]
            
        return res
