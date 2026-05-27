"""任务三：DPO（Direct Preference Optimization）训练脚本。

在 SFT 模型基础上，使用 MOSS-003-dpo 偏好数据继续训练。
需要 reference model（SFT 冻结模型）来计算 implicit reward。
"""

# TODO: 1. 加载 MOSS-003-dpo 偏好数据（chosen/rejected 对）
# TODO: 2. 加载 SFT 后的模型（policy model，从 ckpt/sft/ 恢复 LoRA 权重）
# TODO: 3. 加载 SFT 初始模型作为 reference model（冻结，不更新）
# TODO: 4. 对每个偏好对：
#     - format_messages + tokenize chosen
#     - format_messages + tokenize rejected
#     - 计算 DPO loss：
#       loss = -log(sigmoid(beta * (log_p_chosen - log_p_ref_chosen
#                                   - log_p_rejected + log_p_ref_rejected)))
# TODO: 5. 训练循环（只更新 policy model 的 LoRA 参数）
# TODO: 6. 训练完成后保存 LoRA 权重到 ckpt/dpo/
raise NotImplementedError("TODO: 实现 DPO 训练脚本")
