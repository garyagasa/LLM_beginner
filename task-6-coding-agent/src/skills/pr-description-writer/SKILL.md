---
name: pr-description-writer
description: PR 描述生成 — 根据 git diff 自动生成 Pull Request 描述
---

# PR Description Writer

## 目标

根据代码变更自动生成结构化的 PR 描述。

## 工作流

1. **获取变更**：运行 `git_diff` 获取完整 diff
2. **分析变更**：
   - 识别修改的文件和模块
   - 判断变更类型（feat/fix/refactor/docs/test）
   - 提取关键改动点
3. **生成描述**：
   - **Summary**：1-3 个要点概述变更
   - **Test plan**：建议的测试清单
   - **Breaking changes**：如有不兼容变更则特别标注

## 输出格式

```markdown
## Summary
- ...

## Test plan
- [ ] ...

## Notes
- ...
```

## 工具

- `git_diff`：获取代码变更
- `read_file`：读取文件了解上下文
