import sys; import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from main import get_args_parser
from models.deformable_detr import build
from kn_util.data import collection_to_device
import torch

def model_forward():
    args = get_args_parser().parse_args()
    model, criterion, postprocessors = build(args)
    model = model.cuda()
    B = 16
    H = 224
    W = 224

    inputs = dict(samples=[torch.randn(3, H, W)] * B)
    inputs = collection_to_device(inputs, "cuda")
    out = model(**inputs)
    
if __name__ == "__main__":
    model_forward()