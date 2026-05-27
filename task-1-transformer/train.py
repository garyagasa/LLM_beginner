"""任务一：训练 TransformerClassifier 在 ChnSentiCorp 上做情感分类。

训练完成后将 checkpoint 保存到 ckpt/best.pt。
"""

# TODO: 1. 加载 ChnSentiCorp 数据（data/download.py 下载后得到）
# TODO: 2. 加载 tokenizer（如 bert-base-chinese 的 tokenizer）
# TODO: 3. 构建 DataLoader（tokenize + padding）
# TODO: 4. 初始化 TransformerClassifier
# TODO: 5. 训练循环（AdamW, CrossEntropyLoss, 记录最佳 dev acc）
# TODO: 6. 每个 epoch 后在 dev 集上评估，保存最佳 checkpoint 到 ckpt/best.pt
# TODO: 7. （可选）用 matplotlib 画训练 loss/acc 曲线
# TODO: 8. （可选）抽一个样本画注意力热图
raise NotImplementedError("TODO: 实现训练脚本")
