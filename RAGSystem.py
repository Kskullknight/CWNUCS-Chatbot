import logging
from LLMClient import LLMAPIClient
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import faiss
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from ChatBotGraph import ChatBotGraph
from langchain_core.messages import HumanMessage
from typing import Dict, Any
import asyncio


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class AdvancedSchoolNoticeRAG:
    def __init__(self, data_file_path: str, model_name: str):
        """
        고급 RAG 시스템 초기화 (스트리밍 지원)

        Args:
            data_file_path: 공지사항 데이터 파일 경로
            model_name: 임베딩 모델 이름
        """

        self.data_file_path = data_file_path
        self.model_name = model_name
        
        self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        self.llm_client = LLMAPIClient()

        # 문서 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # 데이터 저장소
        self.documents = []
        self.document_chunks = []
        self.faiss_index = None
        self.chunk_to_doc_map = []

        # 캐시 파일 경로
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.index_cache_path = self.cache_dir / "faiss_index.bin"
        self.docs_cache_path = self.cache_dir / "documents.pkl"
        # self.chunks_cache_path = self.cache_dir / "chunks.pkl"

        # 데이터 로드 및 인덱스 구축
        self._load_and_index_documents()

        # LangGraph 워크플로우 초기화
        self.graph = ChatBotGraph(self.embedding_model, self.faiss_index, self.documents)
        
        
    def _load_and_index_documents(self):
        """문서 로드 및 FAISS 인덱스 구축"""
        
        # 캐시된 인덱스가 있고 데이터 파일이 변경되지 않았다면 캐시 사용
        if self._should_use_cache():
            self._load_from_cache()
            return

        # 문서 로드
        self._load_documents()

        # 문서 청킹
        # self._chunk_documents()

        # 임베딩 생성 및 FAISS 인덱스 구축
        self._build_faiss_index_use_document()

        # 캐시 저장
        self._save_to_cache()
    
    
    def _should_use_cache(self) -> bool:
        """캐시 사용 여부 결정"""
        if not all([
            self.index_cache_path.exists(),
            self.docs_cache_path.exists(),
            # self.chunks_cache_path.exists()
        ]):
            return False

        # 데이터 파일이 캐시보다 최신인지 확인
        try:
            data_mtime = os.path.getmtime(self.data_file_path)
            cache_mtime = os.path.getmtime(self.index_cache_path)
            return cache_mtime > data_mtime
        except:
            return False


    def _load_from_cache(self):
        """캐시에서 데이터 로드"""
        try:
            # FAISS 인덱스 로드
            self.faiss_index = faiss.read_index(str(self.index_cache_path))

            # 문서 및 청크 로드
            with open(self.docs_cache_path, 'rb') as f:
                self.documents = pickle.load(f)

            # with open(self.chunks_cache_path, 'rb') as f:
            #     data = pickle.load(f)
            #     self.document_chunks = data['chunks']
            #     self.chunk_to_doc_map = data['chunk_to_doc_map']
            
            print("Use cache")


        except Exception as e:
            self._load_documents()
            # self._chunk_documents()
            self._build_faiss_index()
            self._save_to_cache()
            

    def _load_documents(self):
        """Excel/CSV 파일에서 문서 로드"""
        try:
            # 파일 확장자에 따라 로딩 방법 선택
            df = pd.read_csv(self.data_file_path, encoding='utf-8')
            # df = pd.read_csv(self.data_file_path, encoding='euc-kr')
            # df = pd.read_csv(self.data_file_path, encoding='cp949')

            for idx, row in df.iterrows():
                # 컬럼명 정규화 (대소문자 및 공백 처리)
                normalized_row = {k.lower().strip(): v for k, v in row.items()}

                metadata = {
                    'view_count': int(normalized_row['view_count']),
                    'notice': bool(normalized_row['notice']),
                    'reg_date': normalized_row['reg_date'],
                    # TODO: url 추가
                }
                
                document = {
                    'doc_id': idx,
                    'title': normalized_row['title'],
                    'full_text': f"{normalized_row['title']}\n\n{normalized_row['content']}",
                    'metadata': metadata
                }

                self.documents.append(document)

        except Exception as e:
            raise
        
        
    def _build_faiss_index_use_document(self):
        """FAISS 인덱스 구축"""

        if not self.documents:
            raise ValueError("청크가 없습니다. 문서를 먼저 로드해주세요.")

        # 모든 청크의 임베딩 생성
        document_texts = [doucment['full_text'] for doucment in self.documents]

        embeddings = self.embedding_model.encode(
            document_texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # FAISS 인덱스 생성 (코사인 유사도 사용)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        # 임베딩 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(embeddings)

        # 인덱스에 추가
        self.faiss_index.add(embeddings.astype(np.float32))
        
        self._save_to_cache()


    def _save_to_cache(self):
        """캐시에 데이터 저장"""
        try:
            # FAISS 인덱스 저장
            faiss.write_index(self.faiss_index, str(self.index_cache_path))

            # 문서 저장
            with open(self.docs_cache_path, 'wb') as f:
                pickle.dump(self.documents, f)

            # 청크 저장
            # with open(self.chunks_cache_path, 'wb') as f:
            #     pickle.dump({
            #         'chunks': self.document_chunks,
            #         'chunk_to_doc_map': self.chunk_to_doc_map
            #     }, f)


        except Exception as e:
            print(f"⚠️ 캐시 저장 실패: {e}")
    
        
    async def answer_question_async(self, query: str) -> Dict[str, Any]:
        """비동기 질의응답 (스트리밍 옵션)"""
        

        # 워크플로우 실행
        result = await self.graph.start_graph(query)

        return {
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_docs"],
            "confidence": result["confidence"],
            "metadata": result["metadata"]
        }
    
    
    def answer_question(self, query: str, streaming: bool = False) -> Dict[str, Any]:
        """동기 질의응답 (스트리밍 옵션)"""
        return asyncio.run(self.answer_question_async(query))
        
if __name__ == "__main__":
    test = AdvancedSchoolNoticeRAG("./output.csv", "Qwen/Qwen3-Embedding-8B")
    while True:
        query = input("질문: ")
        if query == 'exit':
            break
        
        test.answer_question(query)