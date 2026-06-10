# src/ —— 手写 Transformer 实现包
#
# 文件职责：
#   attention.py    M1 缩放点积注意力 + M2 多头注意力
#   block.py        M2 TransformerBlock（attention + FFN + residual + LayerNorm）
#   model.py        M3 TransformerClassifier + load_for_eval 工厂函数
