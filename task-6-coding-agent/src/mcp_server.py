"""任务六：MCP server — 暴露代码操作工具。

接口约定：
  - 可独立运行：python src/mcp_server.py
  - 暴露工具：read_file / write_file / run_tests / git_diff / git_apply

运行方式（MCP Python SDK）：
  python src/mcp_server.py
"""

# TODO: 1. 使用 mcp Python SDK 创建 MCP server
# TODO: 2. 注册以下工具（每个工具有 name, description, inputSchema）：
#
#    read_file(path: str, start_line?: int, end_line?: int) -> str
#      - 读取文件内容，支持行号范围
#
#    write_file(path: str, content: str) -> str
#      - 写入文件（创建或覆盖），返回确认信息
#
#    run_tests(test_path?: str) -> str
#      - 运行 pytest，返回测试结果（stdout + stderr）
#      - 如不指定 path 则运行全部
#
#    git_diff() -> str
#      - 返回 git diff 结果（当前 uncommitted 变更）
#
#    git_apply(patch: str) -> str
#      - 应用给定的 patch 字符串，返回结果
#
# TODO: 3. 每个工具的 handler 实现具体操作
# TODO: 4. 使用 mcp.server.stdio 启动 stdio transport

raise NotImplementedError("TODO: 实现 MCP server")


# 独立的 main 入口
if __name__ == "__main__":
    # TODO: 启动 MCP server（stdio transport）
    raise NotImplementedError("TODO: 启动 MCP server")
