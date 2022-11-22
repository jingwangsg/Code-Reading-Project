python test_model.py --dataset_file cocodown \
    --coco_path data/coco \
    --sample_rate 0.01 \
    --batch_size 4 \
    --model dela-detr \
    --repeat_label 2 \
    --nms \
    --num_queries 300