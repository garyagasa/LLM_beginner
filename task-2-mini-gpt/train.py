"""任务二：mini-GPT 预训练脚本。

next-token prediction, AdamW, cosine schedule, gradient clipping。
"""

# TODO: 1. 加载数据（data/download.py 下载的 poetry / tinystories / skypile）
# TODO: 2. 训练 BPE tokenizer 或加载已有词表，保存到 ckpt/tokenizer.json
# TODO: 3. 构建 DataLoader（tokenize + 固定长度切块 + shift 构造 next-token labels）
# TODO: 4. 初始化 MiniGPT
# TODO: 5. 训练循环：
#     - AdamW optimizer + cosine learning rate schedule
#     - gradient clipping
#     - 每个 step 记录 loss/perplexity
#     - 定期在 dev 集上评估 perplexity
#     - 定期生成样本观察质量
# TODO: 6. 保存最佳 checkpoint 到 ckpt/best.pt
# TODO: 7. （可选）wandb / tensorboard 日志
raise NotImplementedError("TODO: 实现训练脚本")
