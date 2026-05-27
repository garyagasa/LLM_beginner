"""任务六：Skill 加载器。

接口约定（eval/run.py 会自动检测）：
  - class SkillLoader 含 list_skills() -> List[dict], load(name) -> str
"""

import os
from typing import List, Dict


class SkillLoader:
    """扫描 src/skills/*/SKILL.md，按需加载 Skill 内容到 agent context。"""

    def __init__(self, skills_dir: str = "src/skills"):
        """初始化。

        Args:
            skills_dir: skills 目录路径
        """
        self.skills_dir = skills_dir
        # TODO: 扫描 skills_dir 下的所有 SKILL.md 文件
        raise NotImplementedError("TODO: 实现 SkillLoader.__init__")

    def list_skills(self) -> List[Dict]:
        """列出所有可用 Skill。

        Returns:
            List[dict]: 每个元素含 name, description
        """
        # TODO: 1. 遍历 skills_dir 下的每个子目录
        # TODO: 2. 读取每个 SKILL.md 的 frontmatter（name + description）
        # TODO: 3. 返回列表
        raise NotImplementedError("TODO: 实现 list_skills")

    def load(self, name: str) -> str:
        """按名称加载 Skill 的完整内容。

        Args:
            name: Skill 名称

        Returns:
            str: SKILL.md 的完整 markdown 内容
        """
        # TODO: 1. 根据 name 找到对应目录
        # TODO: 2. 读取 SKILL.md 全文
        # TODO: 3. 返回内容
        raise NotImplementedError("TODO: 实现 load")
