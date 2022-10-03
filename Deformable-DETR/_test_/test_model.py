import sys; sys.path.insert(0,"..")
sys.path.insert(0, "../..")
from models.deformable_detr import build
from main import get_args_parser
from my_util.general import black_pprint
from util.misc import NestedTensor
import torch

def __test_detr__(args):
    model, criterion, postprocessors = build(args)
    model = model.cuda()
    mock_x = torch.randn((8, 3, 224, 224)).cuda()
    mock_mask = torch.ones((8, 224, 224), dtype=torch.long).cuda()
    mock_mask[:, :200, :200] = 0

    mock_nested_x = NestedTensor(mock_x, mock_mask)
    o = model(mock_nested_x)
    import ipdb; ipdb.set_trace() #FIXME

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    # args.two_stage=True
    black_pprint(args)
    __test_detr__(args)