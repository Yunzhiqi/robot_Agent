# from utils import config_hander
# print(config_hander.rag_config['chat_model_name'])

# from utils import prompt_loader
# print(prompt_loader.report_prompt+'\n'+prompt_loader.system_prompt)

# from rag.knowledge_service import KnowledgeService
# test=KnowledgeService()
# test.load_doc()
# retriver=test.get_retriever()
# res=retriver.invoke('迷路')
# for r in res:
#     print(r.page_content)
#     print('='*20)

from rag.rag_service import RagSummrizeService
test=RagSummrizeService()
print(test.rag_summarize('小户型适合什么扫地机器人'))