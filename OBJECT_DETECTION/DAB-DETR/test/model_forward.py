import sys; import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from models.dab_deformable_detr.dab_deformable_detr import build_dab_deformable_detr
from util.misc import nested_tensor_from_tensor_list
from main import get_args_parser
import torch
from kn_util.debug import explore_content as EC

args = get_args_parser().parse_args()
model, criterion, postprocessors = build_dab_deformable_detr(args)

B = 4
N = 224
imgs = [torch.randn(3, N, N).cuda()] * B
imgs = nested_tensor_from_tensor_list(imgs)

imgs = imgs
model = model.cuda()
out = model(imgs)

import ipdb; ipdb.set_trace() #FIXME ipdb