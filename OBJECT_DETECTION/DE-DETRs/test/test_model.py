import sys; import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from main import get_args_parser
from kn_util.data import collection_to_device
import torch
from models.detr import build
from einops import repeat

args = get_args_parser().parse_args()
args.sample_rate = 0.01
args.batch_size = 4
args.model = "dela-detr"
args.nms = True
args.num_queries = 300

model, criterion, posprocess = build(args)
model = model.cuda()
B = 16
H = 224
W = 224

img_size = repeat(torch.tensor([H,W]), "i -> b i", b=B)
inputs = dict(samples=[torch.randn(3, H, W)] * B, meta_info=dict(size=img_size))
inputs = collection_to_device(inputs, "cuda")
out = model(**inputs)
import ipdb; ipdb.set_trace() #FIXME