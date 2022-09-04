# Framework

---

# Code Notes

AdaMixer/configs/adamixer/adamixer_r50_1x_coco.py

## 模型配置部分

- QueryBased (Overall Architecture)
- ResNet (backbone)
- ChannelMapping (neck)
- InitialQueryGenerator (rpn_head)
- AdaMixerDecoder (roi_head)

QueryBased继承自SparseRCNN，没有做任何改动

backbone, neck, rpn_head, roi_head是TwoStageDetector的标准结构
