import sys,os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import logging
import torch.nn as nn
from models.cmnext_conf import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config

parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default='./experiments/ec_example_phase2.yaml', help='Yaml experiment file')
parser.add_argument('-ckpt', '--ckpt', type=str, default='./ckpt/early_fusion_detection.pth', help='Checkpoint')
parser.add_argument('-path', '--path', type=str, default='example.png', help='Image path')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

config = update_config(config, args.exp)

loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl)

gpu = args.gpu

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = config.CUDNN.ENABLED

modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

model = CMNeXtWithConf(config.MODEL)

ckpt = torch.load(args.ckpt)

model.load_state_dict(ckpt['state_dict'])
modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

modal_extractor.to(device)
model = model.to(device)
modal_extractor.eval()
model.eval()

class MMF(nn.Module):
    def __init__(self,modal_extractor,model):
        super().__init__()
        self.model=model
        self.modal_extractor=modal_extractor
        self.modal_extractor.eval()
        self.model.eval()
    
    def forward(self, x, originalShape=(512,512)):
        img = np.array(x)
        img=torch.tensor(np.expand_dims(np.transpose(img,(2, 0, 1)), axis=0), dtype=torch.float) / 256.0 #0-1 scale
        x=img.to(device)

        modals = self.modal_extractor(x)

        images_norm = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inp = [images_norm] + modals

        anomaly, confidence, detection = self.model(inp)

        map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()

        det = detection#label
        probability = torch.sigmoid(det).item()
        label=0
        if probability>0.5:
            label=1

        #seg_map = Image.fromarray(map).resize(originalShape)

        return map,label,probability
    
model=MMF(modal_extractor,model)