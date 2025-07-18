import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

class MemoryModule:
    """
    Langchain + Chroma + OpenAIEmbeddings 기반의 메모리 모듈.
    관측(obs), 행동(action), 결과(result) 및 추가 메타데이터를 벡터 DB에 저장/검색.
    """
    def __init__(self, db_path: Optional[str] = None):
    # MY_KEY.txt 파일에서 OpenAI API 키 읽기
        with open("MY_KEY.txt", "r") as f:
            api_key = f.read().strip()  # 양쪽 공백 제거

    # 임베딩 방식 및 DB 경로 설정
        if os.environ.get("OPENAI_API_TYPE", "openai") == 'azure':
            self.embedding = OpenAIEmbeddings(
                deployment=os.environ['AZURE_EMBED_DEPLOY_NAME'],
                chunk_size=1,
                openai_api_key=api_key
            )
        else:
            self.embedding = OpenAIEmbeddings(
                openai_api_key=api_key
            )

        db_path = db_path or os.path.join('./db', 'chroma_memory/')
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
    )
        print(f"[MemoryModule] Loaded {db_path}. Now the database has {len(self.scenario_memory._collection.get(include=['embeddings'])['embeddings'])} items.")

    def save(self, obs: Any, action: Any, result: Any, extra: Dict = None):
        """
        새로운 (관측, 행동, 결과) 사례를 벡터 DB에 저장.
        obs/action/result는 str 또는 dict 등 직렬화 가능한 형태여야 함.
        """
        print(f"[DEBUG][Memory] save called | "
      f"step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane={obs.ego_vehicle_state.lane_id}")

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
        # 중복 체크 (obs가 이미 포함된 경우 업데이트)
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
        """
        현재 obs와 유사한 k개의 사례를 반환 (임베딩 기반 cosine similarity)
        """
        print(f"[DEBUG][Memory] retrieve_similar_cases called "
        f"(step={obs.step_count}, speed={obs.ego_vehicle_state.speed:.2f}, lane={obs.ego_vehicle_state.lane_id})")

        obs_str = str(obs)
        similarity_results = self.scenario_memory.similarity_search_with_score(
            obs_str, k=k)
        fewshot_results = []
        for doc, score in similarity_results:
            fewshot_results.append(doc.metadata)
        print(f"[DEBUG][Memory] retrieve_similar_cases found {len(fewshot_results)} cases")
        print(f"[MemoryModule] Retrieved {len(fewshot_results)} similar cases.")
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