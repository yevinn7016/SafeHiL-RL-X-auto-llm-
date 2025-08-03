# HuggingFaceEmbeddings를 사용하려면 다음 패키지가 필요합니다:
# pip install sentence-transformers

import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
#import requests

# OllamaEmbeddings 클래스는 남겨두되, 사용하지 않도록 주석 처리
# class OllamaEmbeddings:
#     """
#     Ollama 임베딩 모델(nomic-embed-text 등)로 텍스트 임베딩을 생성하는 래퍼 클래스
#     """
#     def __init__(self, model: str = "nomic-embed-text"):
#         self.model = model
#         self.url = "http://localhost:11434/api/embeddings"
# 
#     def embed_documents(self, texts: List[str]) -> List[list]:
#         embeddings = []
#         for text in texts:
#             payload = {"model": self.model, "prompt": text}
#             try:
#                 response = requests.post(self.url, json=payload)
#                 response.raise_for_status()
#                 result = response.json()
#                 emb = result.get("embedding", [])
#                 embeddings.append(emb)
#             except Exception as e:
#                 print(f"[ERROR] Ollama 임베딩 호출 실패: {e}")
#                 embeddings.append([])
#         return embeddings
# 
#     def embed_query(self, text: str) -> list:
#         return self.embed_documents([text])[0]

class MemoryModule:
    """
    Langchain + Chroma + HuggingFaceEmbeddings 기반의 메모리 모듈.
    관측(obs), 행동(action), 결과(result) 및 추가 메타데이터를 벡터 DB에 저장/검색.
    """
    def __init__(self, db_path: Optional[str] = None):
        # HuggingFace 임베딩으로 변경
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        db_path = db_path or os.path.join('./db', 'chroma_memory/')
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )
        print(f"[MemoryModule] Loaded {db_path}. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

    def save(self, obs: Any, action: Any, result: Any, extra: Dict = None):
        #print(f"[DEBUG][Memory] save called | "
      #f"step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane={obs.ego_vehicle_state.lane_id}")

        obs_str = str(obs)
        action_str = str(action)
        result_str = str(result)
        metadata = {
            "obs": obs_str,
            "action": action_str,
            "result": result_str
        }
        if extra:
            metadata.update(extra)
        get_results = self.scenario_memory._collection.get(
            where_document={"$contains": obs_str}
        )
        if len(get_results['ids']) > 0:
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas=metadata
            )
            print(f"[MemoryModule] Modified a memory item. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")
        else:
            doc = Document(
                page_content=obs_str,
                metadata=metadata
            )
            self.scenario_memory.add_documents([doc])
            print(f"[MemoryModule] Added a memory item. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

    def retrieve_similar_cases(self, obs: Any, k: int = 3) -> List[Dict]:
        #print(f"[DEBUG][Memory] retrieve_similar_cases called "
        #f"(step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane={obs.ego_vehicle_state.lane_id})")

        obs_str = str(obs)
        similarity_results = self.scenario_memory.similarity_search_with_score(
            obs_str, k=k)
        fewshot_results = []
        for doc, score in similarity_results:
            fewshot_results.append(doc.metadata)
        #print(f"[DEBUG][Memory] retrieve_similar_cases found {len(fewshot_results)} cases")
        #print(f"[MemoryModule] Retrieved {len(fewshot_results)} similar cases.")
        return fewshot_results

    def deleteMemory(self, ids: List[str]):
        self.scenario_memory._collection.delete(ids=ids)
        print(f"[MemoryModule] Deleted {len(ids)} memory items. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print(f"[MemoryModule] Merge complete. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

if __name__ == "__main__":
    # 간단 테스트
    mem = MemoryModule()
    mem.save({"speed": 10, "lane": 2}, [1, 0], "success")
    mem.retrieve_similar_cases({"speed": 10, "lane": 2}) 