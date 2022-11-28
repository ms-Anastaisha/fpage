from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from .roi_pytorch_impl import roi_tanh_polar_restore, roi_tanh_polar_warp
from .fcn import  FCN
from .resnet import Backbone
from torch.nn.functional import softmax

DECODER_MAP = {
    'fcn': FCN,
}

WEIGHT = {
    'resnet50-fcn-14': (Path(__file__).parent / 'resnet/weights/resnet50-fcn-14.torch', 0.5, 0.5, (513, 513)),
}


class SegmentationModel(nn.Module):

    def __init__(self, encoder='rtnet50', decoder='fcn', num_classes=14):
        super().__init__()

        self.encoder = Backbone(encoder)
        in_channels = self.encoder.num_channels
        self.decoder = DECODER_MAP[decoder.lower()](
            in_channels=in_channels, num_classes=num_classes)
        self.low_level = getattr(self.decoder, 'low_level', False)

    def forward(self, x, rois):
        input_shape = x.shape[-2:]
        features = self.encoder(x, rois)

        low = features['c1']
        high = features['c4']
        if self.low_level:
            x = self.decoder(high, low)
        else:
            x = self.decoder(high)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=False)
        return x


class FaceParser(object):
    def __init__(self, device='cuda:0', ckpt=None, encoder='rtnet50', decoder='fcn', num_classes=11):
        self.device = device
        model_name = '-'.join([encoder, decoder, str(num_classes)])
        assert model_name in WEIGHT, f'{model_name} is not supported'

        pretrained_ckpt, mean, std, sz = WEIGHT[model_name]
        self.sz = sz

        self.model = SegmentationModel(encoder, decoder, num_classes)

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        if ckpt is None:
            ckpt = pretrained_ckpt
        ckpt = torch.load(ckpt, 'cpu')
        ckpt = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(ckpt, True)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def predict_img(self, img, bboxes, rgb=False):

        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            if rgb:
                img = img[:, :, ::-1]
        else:
            raise TypeError
        h, w = img.shape[:2]

        num_faces = len(bboxes)


        imgs = [roi_tanh_polar_warp(img, b, *self.sz, keep_aspect_ratio=True) for b in bboxes]
        imgs = [self.transform(img) for img in imgs]
        bboxes_tensor = torch.tensor(
            bboxes).view(num_faces, -1).to(self.device)

        # img = img.repeat(num_faces, 1, 1, 1)
        # img = roi_tanh_polar_warp(
            # img, bboxes_tensor, target_height=self.sz[0], target_width=self.sz[1], keep_aspect_ratio=True)
        # img = self.transform(img).unsqueeze(0).to(self.device)

        img = torch.stack(imgs).to(self.device)
        logits = self.model(img, bboxes_tensor)
        mask = self.restore_warp(h, w, logits, bboxes_tensor)
        return mask

    def restore_warp(self, h, w, logits: torch.Tensor, bboxes_tensor):
        logits = softmax(logits, 1)
        logits[:, 0] = 1 - logits[:, 0]  # background class
        logits = roi_tanh_polar_restore(
            logits, bboxes_tensor, w, h, keep_aspect_ratio=True
        )
        logits[:, 0] = 1 - logits[:, 0]
        predict = logits.cpu().argmax(1).numpy()
        return predict

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor, rois: torch.Tensor):
        features = self.model.encoder(x, rois, return_features=True)
        x = self.model.decoder(features['c4'])
        features['logits'] = x
        return features
