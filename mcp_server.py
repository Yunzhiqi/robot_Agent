import os
import sys
from mcp.server.fastmcp import FastMCP
import random
import json

# 确保能够正常导入项目内部的 config 和 utils 工具
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.path_tool import get_abs_path
from utils.config_hander import agent_config
from utils.log import logger

# 初始化 MCP Server，命名为 RobotBackend（扫地机器人后端中台）
mcp = FastMCP("RobotBackend")

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
external_data = {}

def generate_external_data():
    """加载本地的 CSV 数据文件以模拟数据库请求"""
    global external_data
    if not external_data:
        external_data_path = get_abs_path(agent_config["external_data_path"])
        if not os.path.exists(external_data_path):
            logger.error(f"[MCP Server] 致命错误：外部数据文件 {external_data_path} 不存在！")
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")

        logger.info(f"[MCP Server] 正在加载外部数据文件: {external_data_path}")
        with open(external_data_path, "r", encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")
                user_id = arr[0].replace('"', "")
                feature = arr[1].replace('"', "")
                efficiency = arr[2].replace('"', "")
                consumables = arr[3].replace('"', "")
                comparison = arr[4].replace('"', "")
                time = arr[5].replace('"', "")

                if user_id not in external_data:
                    external_data[user_id] = {}

                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }
        logger.info(f"[MCP Server] 外部数据加载完成，共加载 {len(external_data)} 个用户数据。")

# ===== 下面是暴露给大模型的动态工具 =====

@mcp.tool()
def get_weather(city: str) -> str:
    """获取指定城市的天气，以消息字符串的形式返回"""
    logger.info(f"[MCP Server] 执行工具: get_weather, 接收参数: city={city}")
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"

@mcp.tool()
def get_user_location() -> str:
    """获取用户所在城市的名称，以纯字符串形式返回"""
    logger.info("[MCP Server] 执行工具: get_user_location")
    return random.choice(["深圳", "合肥", "杭州"])

@mcp.tool()
def get_user_id() -> str:
    """获取用户的ID，以纯字符串形式返回"""
    logger.info("[MCP Server] 执行工具: get_user_id")
    return random.choice(user_ids)

@mcp.tool()
def get_current_month() -> str:
    """获取当前月份，以纯字符串形式返回"""
    logger.info("[MCP Server] 执行工具: get_current_month")
    return random.choice(month_arr)

@mcp.tool()
def fetch_external_data(user_id: str, month: str) -> str:
    """从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回， 如果未检索到返回空字符串"""
    logger.info(f"[MCP Server] 执行工具: fetch_external_data, 接收参数: user_id={user_id}, month={month}")
    generate_external_data()
    try:
        # 转换为 JSON 字符串返回，更符合标准化 API 的响应格式
        return json.dumps(external_data[user_id][month], ensure_ascii=False)
    except KeyError:
        logger.warning(f"[MCP Server] 数据未找到：未能检索到用户 {user_id} 在 {month} 的使用记录数据")
        return ""

if __name__ == "__main__":
    # 启动 stdio 通信模式的服务
    mcp.run()