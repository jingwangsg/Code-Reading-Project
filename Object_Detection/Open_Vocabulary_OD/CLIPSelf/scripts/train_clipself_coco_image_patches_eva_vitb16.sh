DATA_DIR=$HOME/TASKS/OVOD/data
CHECKPOINTS=$HOME/TASKS/OVOD/checkpoints
NPROC=$1
torchrun --nproc_per_node $NPROC -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
    --model EVA02-CLIP-B-16 --pretrained eva --warmup 1000 --zeroshot-frequency 1 --dataset-type grid_distill \
    --test-type coco_panoptic --train-data $DATA_DIR/coco/annotations/instances_train2017.json \
    --val-data $DATA_DIR/coco/annotations/panoptic_val2017.json \
    --embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTB16.npy \
    --train-image-root $DATA_DIR/coco/train2017 \
    --val-image-root $DATA_DIR/coco/val2017 \
    --val-segm-root $DATA_DIR/coco/annotations/panoptic_val2017 \
    --cache-dir $CHECKPOINTS/EVA02_CLIP_B_psz16_s8B.pt --log-every-n-steps 50 \
    --lock-image --save-frequency 6 --lock-image-unlocked-groups 12 --extract-type="v2" \
    --name clipself_coco_6_save6_test1_eva_vitb16_12layers --downsample-factor 16 --det-image-size 1024 \
    --alpha 0.7
