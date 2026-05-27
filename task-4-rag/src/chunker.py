"""任务四：文档分块（chunking）。

接口约定（eval/run.py 会自动检测）：
  - chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> List[str]:
    """将长文本切分为重叠的 chunk。

    支持三种策略：
    - 固定大小切分（按字符/token）
    - 递归切分（按段落 → 句子 → 词）
    - 语义切分（按 embedding 相似度断点）

    Args:
        text: 输入文本
        chunk_size: 每个 chunk 的目标大小
        overlap: chunk 之间的重叠大小

    Returns:
        List[str]: 切分后的文本块列表
    """
    # TODO: 实现至少一种 chunking 策略
    # 1. 固定大小：按 chunk_size 滑动窗口切分，overlap 重叠
    # 2. (可选) 递归：先按 \\n\\n 再按 \\n 再按 。 分
    # 3. (可选) 语义：用 embedding 找语义断点
    raise NotImplementedError("TODO: 实现 chunk_text")


def extract_text_from_pdf(pdf_path: str) -> str:
    """从 PDF 文件提取纯文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        str: 提取的纯文本
    """
    # TODO: 使用 pypdf 提取文本
    raise NotImplementedError("TODO: 实现 extract_text_from_pdf")
