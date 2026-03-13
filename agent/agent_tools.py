from langchain_core.tools import tool
from rag.rag_service import RagSummrizeService

@tool(description='从向量数据库中检索参考资料')
def rag_summarize(quary:str)->str:
    rag=RagSummrizeService()
    return rag.rag_summarize(quary)

@tool(description='获取指定城市天气,以字符串形式返回')
def get_weather(city:str):
    