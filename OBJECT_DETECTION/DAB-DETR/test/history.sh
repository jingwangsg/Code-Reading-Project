python test/model_forward.py -m dab_deformable_detr \
  --output_dir logs/dab_deformable_detr/R50 \
  --batch_size 2 \
  --coco_path /path/to/your/COCODIR \
  --resume /path/to/our/checkpoint \
  --transformer_activation relu