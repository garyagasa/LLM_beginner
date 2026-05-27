"""任务三：SFT（监督微调）训练脚本。

在 Qwen2.5-0.5B 上使用 MOSS-003-sft 数据进行指令微调。
使用手写 LoRA 做参数高效微调。
"""

# TODO: 1. 加载 MOSS-003-sft 数据（data/download.py 下载）
# TODO: 2. 加载 Qwen2.5-0.5B 模型 + tokenizer
# TODO: 3. 调用 src/lora.py 的 inject_lora 注入 LoRA
# TODO: 4. 对每个对话样本：
#     - format_messages(messages) 格式化
#     - tokenize
#     - build_labels 做 loss masking
# TODO: 5. DataLoader + 训练循环（AdamW, 只更新 LoRA 参数）
# TODO: 6. 训练完成后保存 LoRA 权重到 ckpt/sft/
raise NotImplementedError("TODO: 实现 SFT 训练脚本")
