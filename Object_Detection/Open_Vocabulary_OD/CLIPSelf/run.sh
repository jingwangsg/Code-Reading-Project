python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=1 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=4 \
    --master_port=10827 \
    F-ViT/train.py \
    F-ViT/configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py \
    --seed 0 \
    --launcher pytorch

torchrun --nproc_per_node=4 F-ViT/train.py \
    F-ViT/configs/ov_coco/fvit_vitb16_upsample_fpn_bs64_3e_ovcoco_eva_original.py \
    --seed 0 \
    --launcher pytorch
