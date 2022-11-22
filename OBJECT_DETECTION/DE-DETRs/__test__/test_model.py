import sys; import os.path as osp; sys.path.insert(0, osp.join(osp.dirname(__file__), ".."))
from main import get_args_parser

args = get_args_parser()
args.sample_rate = 0.01
args.batch_size = 4
args.model = "dela-detr"
args.nms = True
args.num_queries = 300

from models.detr import build

model, criterion, posprocess = build(args)
import ipdb; ipdb.set_trace() #FIXME