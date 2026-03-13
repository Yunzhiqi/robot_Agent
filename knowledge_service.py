## 已弃用
# import os
# import hashlib
# from datetime import datetime

# from langchain_chroma import Chroma
# from langchain_community.embeddings import DashScopeEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# import config



# def check_md5(md5_str):
#     if not os.path.exists(config.md5_data_path):
#         return False
    
#     for line in open(config.md5_data_path,'r',encoding='utf-8').readlines():
#         line=line.strip()
#         if md5_str==line:
#             return True
#     return False
# def save_md5(md5_str):
#     if not os.path.exists(config.md5_data_path):
#         open(config.md5_data_path,'w',encoding='utf-8').close()
#     with open(config.md5_data_path,'a',encoding='utf-8') as f:
#         f.write(md5_str+'\n')
# def get_md5(input_str:str,encoding='utf-8'):
#     str_bytes=input_str.encode(encoding=encoding)
    
#     md5_obj=hashlib.md5()
#     md5_obj.update(str_bytes)
#     md5_hex=md5_obj.hexdigest()
    
#     return md5_hex


# class KnowledgeBaseService(object):
    
#     def __init__(self):
#         self.chroma=Chroma(
#             collection_name=config.collection_name,
#             embedding_function=DashScopeEmbeddings(model='text-embedding-v4'),
#             persist_directory=config.persist_path
#         )
        
#         self.spliter=RecursiveCharacterTextSplitter(
#             separators=config.seperators
#         )
#         return
    
#     def upload_str(self,data:str,filename:str):
#         md5_hex=get_md5(data)
#         if check_md5(md5_hex):
#             return '[失败]已有,跳过'
#         knowledge_data=self.spliter.split_text(data)
#         meta_data={
#             'source':filename,
#             'create_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }    
#         self.chroma.add_texts(
#             knowledge_data,
#             metadatas=[meta_data for _ in knowledge_data]
#         )
#         save_md5(md5_hex)
#         return '[成功]已写入文件'+filename
    
# if __name__=='__main__':
#     service=KnowledgeBaseService()
#     ans=service.upload_str('ABC','test')
#     print(ans)