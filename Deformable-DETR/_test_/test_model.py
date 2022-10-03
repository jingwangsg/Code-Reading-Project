import sys; sys.path.insert(0,"..")
from models.deformable_detr import build
from main import get_args_parser

def __test_detr__(args):
    detr = build(args)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    __test_detr__(args)