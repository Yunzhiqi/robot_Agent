import sys
import asyncio
import threading
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from langchain_mcp_adapters.tools import load_mcp_tools

class UniversalSyncMCPClient:
    """
    通用的同步 MCP 客户端。
    用于在同步运行框架（如 LangGraph）中无缝调用基于异步的 MCP 工具。
    """
    def __init__(self, server_script_path: str):
        """
        :param server_script_path: 本地 MCP Server 脚本的绝对路径 (例如: "/path/to/mcp_server.py")
        """
        self.server_script_path = server_script_path
        # 创建一个独立的事件循环并在后台线程中永久运行
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        
        self.tools = []
        self.session = None
        self.stdio_context = None
        self._init_sync()

    def _init_sync(self):
        async def init_mcp():
            # 使用传入的脚本路径启动 MCP Server
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_script_path]
            )
            self.stdio_context = stdio_client(server_params)
            self.read, self.write = await self.stdio_context.__aenter__()
            self.session = ClientSession(self.read, self.write)
            await self.session.__aenter__()
            await self.session.initialize()
            
            # 动态拉取工具
            mcp_tools = await load_mcp_tools(self.session)
            
            # 异步转同步的魔法包装
            for t in mcp_tools:
                def create_sync_run(async_tool):
                    # 显式声明接收 config 和 run_manager
                    def _sync_run(*args, config=None, run_manager=None, **kwargs):
                        if config is None:
                            config = {}
                        return asyncio.run_coroutine_threadsafe(
                            # 传入 config，同时强制设置 run_manager=None 避免异步方法遇到同步管理器报错
                            async_tool._arun(*args, config=config, run_manager=None, **kwargs), self.loop
                        ).result()
                    return _sync_run
                t._run = create_sync_run(t)
            return mcp_tools

        # 阻塞等待后台线程把工具拉取完毕
        future = asyncio.run_coroutine_threadsafe(init_mcp(), self.loop)
        self.tools = future.result()

    def close(self):
        """
        优雅关闭连接的方法（可选），用于清理后台线程和会话
        """
        async def _close():
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self.stdio_context:
                await self.stdio_context.__aexit__(None, None, None)
                
        asyncio.run_coroutine_threadsafe(_close(), self.loop).result()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=2.0)
