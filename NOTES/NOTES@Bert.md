## ./transformers/src/transformers/models/bert/modeling_bert.py

BertModel: Bert模型主类

BertEncoder: 主要实现N层BertLayer

BertLayer: Attention (Encoder模式) OR Attention + Cross Attention (Decoder模式)

Output类(BertOutput, BertSelfOutput): 修饰输出的残差链接+layernorm

Intermediate类(BertIntermediate): 作为输出至Output前的hidden layer (Linear + act)
