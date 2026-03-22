from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphRecursionError
import os
from utils.path_tool import get_abs_path
from utils.config_hander import system_config

from utils.model_factory import chat_model
from utils.prompt_loader import system_prompt, report_prompt
from utils.log import logger
from agent.tools.agent_tools import (rag_summarize, fill_context_for_report, transfer_to_human)

# import sys
# import asyncio
# import threading
# from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp.client.session import ClientSession
# from langchain_mcp_adapters.tools import load_mcp_tools
from utils.sync_mcp_server import UniversalSyncMCPClient

# ==========================================
# 新增：同步化运行 MCP Client 的后台管理器
# 作用：在后台线程维护与 MCP Server 的长连接，并提供兼容 LangGraph 的同步工具
# ==========================================
# class SyncMCPClient:
#     def __init__(self):
#         # 创建一个独立的事件循环并在后台线程中永久运行
#         self.loop = asyncio.new_event_loop()
#         self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
#         self.thread.start()
#         self.tools = []
#         self._init_sync()

#     def _init_sync(self):
#         async def init_mcp():
#             # 通过标准输入输出（stdio）启动并连接我们刚才写的 mcp_server.py
#             server_params = StdioServerParameters(
#                 command=sys.executable,
#                 args=[get_abs_path("mcp_server.py")]
#             )
#             self.stdio_context = stdio_client(server_params)
#             self.read, self.write = await self.stdio_context.__aenter__()
#             self.session = ClientSession(self.read, self.write)
#             await self.session.__aenter__()
#             await self.session.initialize()
            
#             # 使用 LangChain 的 MCP 适配器，动态拉取服务器上的所有工具
#             mcp_tools = await load_mcp_tools(self.session)
            
#             # 核心魔法：为每个异步工具动态绑定同步调用方法，完美骗过同步运行的 LangGraph
#             for t in mcp_tools:
#                 def create_sync_run(async_tool):
#                     def _sync_run(*args, **kwargs):
#                         return asyncio.run_coroutine_threadsafe(
#                             async_tool._arun(*args, **kwargs), self.loop
#                         ).result()
#                     return _sync_run
#                 t._run = create_sync_run(t)
#             return mcp_tools

#         # 阻塞等待后台线程把工具拉取完毕
#         future = asyncio.run_coroutine_threadsafe(init_mcp(), self.loop)
#         self.tools = future.result()


# 1. 定义图的状态 (State)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_report: bool  # 替代原先在 request.runtime.context 中保存的状态
    human_mode: bool # 记录是否处于人工接管模式


class ReactAgent:
    def __init__(self):
        # 1. 启动并连接本地的 MCP Server
        logger.info("[MCP] 正在连接 MCP Server 并动态获取外部工具...")
        self.mcp_client = UniversalSyncMCPClient(get_abs_path(system_config["mcp_server_path"]))
        # self.mcp_client = SyncMCPClient()
        
        mcp_tools = self.mcp_client.tools
        logger.info(f"[MCP] 成功拉取到 {len(mcp_tools)} 个 MCP 外部工具！")
        
        # 2. 将本地特化的“大脑工具”与 MCP 的“外部四肢”混合编队
        self.tools = [rag_summarize, fill_context_for_report, transfer_to_human] + mcp_tools

        # 为模型绑定工具能力
        self.model = chat_model.bind_tools(self.tools)

        # 2. 初始化图，指定状态结构
        workflow = StateGraph(AgentState)

        # 3. 添加节点
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.call_tools)
        workflow.add_node("human", self.human_node)

        # 4. 定义流程流转规则 (Edges)
        # 起点路由：根据是否处于人工模式，决定去 AI 节点还是直接去人工节点
        workflow.add_conditional_edges(START, self.route_from_start)
        # tools_condition 是 LangGraph 内置条件：
        # - 若模型回复要求调工具，则流转到 "tools" 节点
        # - 若模型给出最终回答，则流转到 END 结束对话
        workflow.add_conditional_edges("agent", tools_condition)
        
        # 添加条件路由：工具执行完毕后，检查是否呼叫了人工
        workflow.add_conditional_edges("tools", self.route_after_tools, {"human": "human", "agent": "agent"})
        
        # 人工节点执行完毕后，当前流转结束，等待下一次用户输入
        workflow.add_edge("human", END)

        # 5. 编译图，引入 SQLite 数据库记忆存储，跨进程共享状态
        db_path = get_abs_path(system_config["chat_state_db_path"])
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # check_same_thread=False 允许不同进程/线程安全访问同一数据库文件
        # self.conn = sqlite3.connect("chat_state.db", check_same_thread=False)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        
        self.app = workflow.compile(checkpointer=self.memory, interrupt_before=["human"])

    def call_model(self, state: AgentState):
        """大模型思考节点（接管了原本的 log_before_model 和 report_prompt_switch）"""
        messages = state.get("messages", [])
        is_report = state.get("is_report", False)

        logger.info(f"[Agent Node]即将调用模型，当前对话共有 {len(messages)} 条消息。")

        # 动态切换 Prompt (状态机控制人设)
        current_prompt = report_prompt if is_report else system_prompt
        sys_msg = SystemMessage(content=current_prompt)

        # 调用模型 (加上系统提示词和历史消息)
        response = self.model.invoke([sys_msg] + messages)
        
        return {"messages": [response]}

    def call_tools(self, state: AgentState):
        """工具执行节点（接管了原本的 monitor_tool）"""
        messages = state.get("messages", [])
        last_message = messages[-1]

        # 执行所有工具调用请求
        tool_node = ToolNode(self.tools)
        result = tool_node.invoke(state)

        # 核心修复：LangChain 的 MCP 适配器默认返回的工具结果是 List 格式
        # 但 DeepSeek 等模型 API 严格要求 ToolMessage 的 content 必须是纯字符串 (string)
        # 所以在这里进行一次拦截清洗，把 list 强制降维转换成 string
        for msg in result.get("messages", []):
            if isinstance(msg.content, list):
                # 提取 MCP 返回的 text 内容并拼接
                msg.content = "\n".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in msg.content
                )

        # 业务逻辑：检查刚才是否调用了生成报告的前置工具
        is_report = state.get("is_report", False)
        human_mode = state.get("human_mode", False)
        for tool_call in last_message.tool_calls:
            logger.info(f"[Tool Node]执行工具：{tool_call['name']}, 参数：{tool_call['args']}")
            if tool_call["name"] == "fill_context_for_report":
                is_report = True  # 核心：改变图状态
            if tool_call["name"] == "transfer_to_human":
                human_mode = True # 开启人工模式标志位

        return {"messages": result["messages"], "is_report": is_report, "human_mode": human_mode}

    def human_node(self, state: AgentState):
        """人工介入节点：实际上是一个占位符。图在此处因为 interrupt_before 会被自动挂起。"""
        logger.info("[Human Node] 流程已暂停，等待人工客服介入。")
        return None

    def route_from_start(self, state: AgentState) -> str:
        """判断每次收到新消息时，应该交给大模型还是直接给人工"""
        if state.get("human_mode", False):
            return "human"
        return "agent"

    def route_after_tools(self, state: AgentState) -> str:
        """判断刚刚执行的工具中，是否包含了转接人工的请求"""
        messages = state.get("messages", [])
        # ToolNode 返回的消息类型为 ToolMessage，其 name 属性记录了刚刚运行的工具名
        if messages and hasattr(messages[-1], "name") and messages[-1].name == "transfer_to_human":
            return "human"
        return "agent"

    def execute_stream(self, query: str, thread_id: str = "user_001"):
        inputs = {
            "messages": [("user", query)],
            "is_report": False
        }
        
        # 使用 Checkpointer 时，必须传入 thread_id 进行会话多线程隔离
        # recursion_limit 用于限制图的最大执行步数，防止大模型无视 Prompt 陷入死循环
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 15
        }
        
        # 将 config 传入 stream 中
        try:
            # 如果当前是人工模式，直接运行图（会被挂起），然后给用户提示
            current_state = self.app.get_state(config)
            if current_state and current_state.values.get("human_mode", False):
                for _ in self.app.stream(inputs, config=config, stream_mode="messages"):
                    pass
                yield "\n\n【系统提示】消息已送达，人工客服正在为您处理中，请稍候..."
                return

            for msg, metadata in self.app.stream(inputs, config=config, stream_mode="messages"):
                # 如果是模型节点生成的文字内容（而非工具调用标识），则将其通过 yield 抛出给前端
                if msg.content and metadata.get("langgraph_node") == "agent":
                    yield msg.content
        except GraphRecursionError:
            logger.error("[Graph Error] 触发最大递归限制，大模型可能陷入死循环。")
            yield "\n\n【系统提示】抱歉，我遇到了一些复杂的问题，思考过程陷入了循环。请您换一种方式提问，或要求转接人工客服。"
