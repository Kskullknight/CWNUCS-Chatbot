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
#이건 로컬에서 썼던건데 원격으로 둬서 그냥 주석 처리함
# from sentence_transformers import SentenceTransformer

#pip install rank-bm25
from rank_bm25 import BM25Okapi

#테스트할땐 GPU부분 주석으로 했었어
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
        
        self.embedding_model = SentenceTransformer(self.model_name)
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

#bm25 인덱스=> 제목,내용 기준으로 한거
        self.merged_corpus_tokens = [
            (doc['title'] + " " + doc['full_text']).split()
            for doc in self.documents
        ]
        self.bm25_merged_index = BM25Okapi(self.merged_corpus_tokens)

        # 문서 청킹
        # self._chunk_documents()

        # 임베딩 생성 및 FAISS 인덱스 구축
        self._build_faiss_index_use_document()

        # 캐시 저장
        self._save_to_cache()

#사용자가 입력한 질문에 포함된 키워드로 제목이랑 내용에 키워드가 모두 포함된 문서들만 골라서 return함
    def filter_documents_by_keyword_inclusion(self, query: str) -> list:
        keywords = query.strip().split() #공백기준

        filtered_docs = []
        for doc in self.documents:
            text = doc['title'] + " " + doc['full_text']

            # 모든 키워드가 있어야 필터링됨 -> any:하나라도 포함되면 통과시키고 싶으면=근데 all하는게 좋음 any는 상관없는거 까지 다 포함시켜서
            if all(keyword in text for keyword in keywords):
                filtered_docs.append(doc)

        print(f"\n[단순 키워드 포함 필터링 결과: {query}]")
        for doc in filtered_docs:
            print(f" - {doc['title']}")

        return filtered_docs
    
    #필터링된 문서들로 FAISS 인덱스 새로 생성
    def build_faiss_index_from_filtered_documents(self, filtered_documents):
        if not filtered_documents:
            raise ValueError("필터링된 문서가 없습니다.")

        #이게 제목+내용
        document_texts = [
        doc['title'] + "\n\n" + doc['full_text']
        for doc in filtered_documents
    ]

#임베딩 벡터로: sentence_transformers 모델(로컬)-> 원격에서 돌릴꺼면 qwen 이겠지?
        embeddings = self.embedding_model.encode(
            document_texts,
            batch_size=8,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))

        return index, filtered_documents

    
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


            self.merged_corpus_tokens = [
                # doc['full_text'].split() #이게 내용가지고 토큰 만드는거고
                (doc['title'] + " " + doc['full_text']).split() #이게 제목+내용으로 토큰만드는거
                for doc in self.documents
            ]
            self.bm25_merged_index = BM25Okapi(self.merged_corpus_tokens)


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
        document_texts = [
        doc['title'] + "\n\n" + doc['full_text']
        for doc in self.documents
        ]
        #밑에껀 니가 해놓은거라 안 지웠음
        # document_texts = [doucment['full_text'] for doucment in self.documents]

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
        filtered_docs = self.filter_documents_by_keyword_inclusion(query)
        #필터링된 문서로 FAISS 인덱스 재구성
        faiss_index, limited_documents = self.build_faiss_index_from_filtered_documents(filtered_docs)
        #ChatBotGraph에 전달
        self.graph = ChatBotGraph(self.embedding_model, faiss_index, limited_documents)

        return asyncio.run(self.answer_question_async(query))
        
if __name__ == "__main__":
    test = AdvancedSchoolNoticeRAG("./output.csv", "Qwen/Qwen3-Embedding-8B")
    #로컬에서 내가 쓴건 그냥 all-mpnet-base-v2 모델 
    while True:
        query = input("질문: ")
        if query == 'exit':
            break
        
        test.answer_question(query)