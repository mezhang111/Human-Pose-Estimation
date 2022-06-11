import torch.nn as nn

from .backbone import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .head import SMPLHead
from .head.model_head2 import ModelHead2

# Factors that can be changed : training epochs, resnet variant used as backbone, architecture of the model_head


class Model(nn.Module):
    def __init__(self, img_res=224):
        super(Model, self).__init__()
        backbone = 'resnet152'
        self.backbone = eval(backbone)(pretrained=True)
        self.use_cam_feats = False
        self.head = ModelHead2(num_input_features=512 if backbone in ['resnet18', 'resnet30'] else 2048)
        self.smpl = SMPLHead(focal_length=5000., img_res=img_res)

    def forward(self, images):
        features = self.backbone(images)
        custom_output = self.head(features)
        smpl_output = self.smpl(rotmat=custom_output['pred_pose'], shape=custom_output['pred_shape'], cam=custom_output['pred_cam'], normalize_joints2d=True)
        smpl_output.update(custom_output)
        return smpl_output
