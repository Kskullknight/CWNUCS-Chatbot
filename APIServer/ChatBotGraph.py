from typing import List, Dict, Any, TypedDict, Annotated, Iterator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
import re
import faiss
import numpy as np
import math
from collections import Counter
from LLMClient import LLMAPIClient
from StreamingDisplayManager import StreamingDisplayManager
from langchain_core.messages import HumanMessage


# 상태 정의
class ChatbotState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str
    confidence: float
    metadata: Dict[str, Any]
    streaming: bool  # 스트리밍 모드 플래그

class ChatBotGraph:
    def __init__(self, embedding_model, faiss_index, documents, threshold=0.5):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.documents = documents
        self.threshold = threshold
        self.llm_client = LLMAPIClient()
        self.streamingDisplayManager = StreamingDisplayManager()
        
        # BM25 관련 파라미터
        self.k1 = 1.2  # 용어 빈도 포화 파라미터
        self.b = 0.75  # 길이 정규화 파라미터
        self.epsilon = 0.25  # IDF 하한값
        
        # 문서 전처리 및 BM25 초기화
        self._preprocess_documents_for_bm25()
        
        self.workflow = self._setup_workflow()
    
    async def start_graph(self, query):
        print(f"[ChatBotGraph] 워크플로우 시작: {query}")
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "metadata": {},
        }
        
        try:
            result = await self.workflow.ainvoke(initial_state)
            print(f"[ChatBotGraph] 워크플로우 완료")
        except Exception as e:
            print(f"[ChatBotGraph] 워크플로우 오류: {e}")
            result = initial_state
            result['answer'] = f"처리 중 오류가 발생했습니다: {e}"
        
        return result
        
    
    def _check_retrieval_docs(self, state: ChatbotState) -> str:
        """검색된 문서가 있는지 확인하여 다음 단계를 결정"""
        retrieved_docs = state.get("retrieved_docs", [])
        
        if not retrieved_docs or len(retrieved_docs) == 0:
            return "no_documents_found"
        else:
            return "context_preparation"
    
    
    def _setup_workflow(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(ChatbotState)

        # 노드 추가
        workflow.add_node("query_processing", self._process_query_node)
        workflow.add_node("document_retrieval", self._retrieve_documents_node)
        workflow.add_node("context_preparation", self._prepare_context_node)
        workflow.add_node("streaming_answer_generation", self._generate_streaming_answer_node)
        workflow.add_node("response_validation", self._validate_response_node)
        workflow.add_node("no_documents_found", self._handle_no_documents)

        # 엣지 추가
        workflow.set_entry_point("query_processing")
        workflow.add_edge("query_processing", "document_retrieval")
        
        workflow.add_conditional_edges(
            source="document_retrieval",
            path=self._check_retrieval_docs,
            path_map={
                "context_preparation": "context_preparation",
                "no_documents_found": "no_documents_found"
            }
        )
        
        workflow.add_edge("context_preparation", "streaming_answer_generation")
        workflow.add_edge("streaming_answer_generation", "response_validation")
        
        
        workflow.add_edge("response_validation", END)
        
        workflow.add_edge("no_documents_found", END)

        return workflow.compile()


    def _handle_no_documents(self, state: ChatbotState) -> ChatbotState:
        """문서를 찾을 수 없을 때 처리"""
        print(f"[ChatBotGraph] 노드 도착: no_documents_found")
        print(f"[ChatBotGraph] 검색 결과 없음")
        state["answer"] = "죄송합니다. 질문과 관련된 공지사항을 찾을 수 없습니다. 다른 키워드로 다시 질문해 주시거나, 창원대학교 홈페이지를 직접 확인해 주세요."
        state["context"] = "관련 문서를 찾을 수 없습니다."
        state["confidence"] = 0.0
        state["metadata"]["no_documents_reason"] = "검색 결과 없음"
        
        self.streamingDisplayManager.update(state["answer"] + "\n")
        
        return state
    
    
    def _process_query_node(self, state: ChatbotState) -> ChatbotState:
        """쿼리 전처리 노드"""
        print(f"[ChatBotGraph] 노드 도착: query_processing")
        query = state.get("query", "")
        print(f"[ChatBotGraph] 원본 쿼리: {query}")

        # 쿼리 정리 및 확장
        processed_query = self._preprocess_query(query)
        print(f"[ChatBotGraph] 처리된 쿼리: {processed_query}")

        state["query"] = processed_query
        state["metadata"] = {"original_query": query, "processed_query": processed_query}

        return state


    def _preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 기본 정리
        query = query.strip()

        # 일반적인 질문 패턴 확장
        query_expansions = {
            r"언제.*신청": "신청 기간 신청일정 신청기간",
            r"어떻게.*신청": "신청 방법 신청절차",
            r"장학금": "장학금 지원금 학비지원",
            r"행사.*일정": "행사 이벤트 프로그램 일정 날짜",
            r"연락처": "연락처 문의 전화 이메일",
            r"등록금": "등록금 학비 납입",
            r"수강신청": "수강신청 강의 과목"
        }

        expanded_query = query
        for pattern, expansion in query_expansions.items():
            if re.search(pattern, query):
                expanded_query += f" {expansion}"

        return query


    def _preprocess_documents_for_bm25(self):
        """BM25를 위한 문서 전처리"""
        self.doc_tokens = []
        self.doc_lengths = []
        self.term_frequencies = []
        
        for doc in self.documents:
            # 제목과 내용을 합쳐서 토큰화
            doc_text = (doc['title'] + ' ' + doc['full_text']).lower()
            tokens = re.findall(r'\b\w+\b', doc_text)
            
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # 용어 빈도 계산
            tf = Counter(tokens)
            self.term_frequencies.append(tf)
        
        # 평균 문서 길이 계산
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        # 문서 빈도 계산 (각 용어가 나타나는 문서 수)
        self.doc_frequencies = Counter()
        for tf in self.term_frequencies:
            for term in tf.keys():
                self.doc_frequencies[term] += 1
        
        # 총 문서 수
        self.total_docs = len(self.documents)


    def _calculate_bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """BM25 점수 계산"""
        doc_tf = self.term_frequencies[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        
        score = 0.0
        
        for term in query_tokens:
            if term in doc_tf:
                # 용어 빈도 (TF)
                tf = doc_tf[term]
                
                # 문서 빈도 (DF)
                df = self.doc_frequencies[term]
                
                # IDF 계산
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5))
                
                # IDF 하한값 적용
                if idf < self.epsilon:
                    idf = self.epsilon
                
                # BM25 점수 계산
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score


    def _bm25_search(self, query: str, top_k: int = 50) -> List[tuple]:
        """BM25를 사용한 문서 검색"""
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        
        if not query_tokens:
            return [(idx, 0.0) for idx in range(len(self.documents))]
        
        # 각 문서에 대해 BM25 점수 계산
        scores = []
        for doc_idx in range(len(self.documents)):
            score = self._calculate_bm25_score(query_tokens, doc_idx)
            scores.append((doc_idx, score))
        
        # 점수 기준 정렬
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 반환 (점수가 0보다 큰 것만)
        return [(idx, score) for idx, score in scores[:top_k] if score > 0]


    def _keyword_search(self, query: str) -> List[int]:
        """BM25 기반 키워드 검색으로 문서 후보 필터링"""
        bm25_results = self._bm25_search(query, top_k=50)
        
        if not bm25_results:
            return list(range(len(self.documents)))
        
        candidate_indices = [idx for idx, score in bm25_results]
        
        print(f"[ChatBotGraph] BM25 검색 결과: {len(candidate_indices)}개 문서")
        for idx, score in bm25_results[:5]:  # 상위 5개만 출력
            print(f"[ChatBotGraph] 문서 {idx}: {self.documents[idx]['title'][:50]}... (BM25 점수: {score:.3f})")
        
        return candidate_indices


    def _retrieve_documents_node(self, state: ChatbotState) -> ChatbotState:
        """하이브리드 문서 검색 노드 (키워드 + 벡터 검색)"""
        print(f"[ChatBotGraph] 노드 도착: document_retrieval")
        query = state["query"]
        print(f"[ChatBotGraph] 검색 쿼리: {query}")

        try:
            # 1단계: 키워드 기반 검색으로 후보 문서 필터링
            keyword_candidates = self._keyword_search(query)
            
            # 키워드 필터링 결과가 너무 많으면 상위 N개만 사용
            if len(keyword_candidates) > 50:
                keyword_candidates = keyword_candidates[:50]
            
            # 2단계: 필터링된 문서들에 대해서만 벡터 검색 수행
            if not keyword_candidates:
                state["retrieved_docs"] = []
                return state
            
            # 쿼리 임베딩 생성
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)

            # 후보 문서들의 임베딩을 추출하여 임시 인덱스 생성
            candidate_embeddings = []
            candidate_docs = []
            
            for idx in keyword_candidates:
                if idx < len(self.documents):
                    candidate_docs.append(self.documents[idx])
                    # 문서 임베딩을 FAISS 인덱스에서 추출
                    doc_embedding = self.faiss_index.reconstruct(idx)
                    candidate_embeddings.append(doc_embedding)
            
            if not candidate_embeddings:
                state["retrieved_docs"] = []
                return state
            
            # 임시 FAISS 인덱스 생성
            candidate_embeddings = np.array(candidate_embeddings).astype(np.float32)
            temp_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
            temp_index.add(candidate_embeddings)
            
            # 벡터 검색 수행
            k = min(10, temp_index.ntotal)
            scores, indices = temp_index.search(query_embedding.astype(np.float32), k)

            # 검색 결과 정리
            retrieved_docs = []
            seen_doc_ids = set()

            for score, temp_idx in zip(scores[0], indices[0]):
                if temp_idx == -1 or temp_idx >= len(candidate_docs):
                    continue

                candidate_doc = candidate_docs[temp_idx]
                doc_id = candidate_doc['doc_id']

                # 중복 문서 제거 (최고 점수만 유지)
                if doc_id not in seen_doc_ids and doc_id < len(self.documents):
                    original_doc = self.documents[doc_id]

                    retrieved_doc = {
                        'doc_id': doc_id,
                        'title': original_doc['title'],
                        'content': original_doc['full_text'],
                        'metadata': original_doc['metadata'],
                        'similarity_score': float(score)
                    }

                    if retrieved_doc['similarity_score'] < self.threshold:
                        continue
                    retrieved_docs.append(retrieved_doc)
                    seen_doc_ids.add(doc_id)

            # 점수 기준 정렬 및 상위 문서 선택
            retrieved_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            state["retrieved_docs"] = retrieved_docs[:5]  # 상위 5개 문서
            
            print(f"[ChatBotGraph] 키워드 검색 후보: {len(keyword_candidates)}개, 최종 선택: {len(retrieved_docs)}개")
            for doc in retrieved_docs:
                print(f"[ChatBotGraph] 문서 ID: {doc['doc_id']}, 제목: {doc['title']}, 유사도: {doc['similarity_score']:.3f}")

        except Exception as e:
            print(f"[ChatBotGraph] 하이브리드 검색 중 에러 발생: {e}")
            state["retrieved_docs"] = []

        return state
    
    
    def _prepare_context_node(self, state: ChatbotState) -> ChatbotState:
        """컨텍스트 준비 노드"""
        print(f"[ChatBotGraph] 노드 도착: context_preparation")
        retrieved_docs = state["retrieved_docs"]
        print(f"[ChatBotGraph] 검색된 문서 수: {len(retrieved_docs)}")

        if not retrieved_docs:
            state["context"] = "관련 문서를 찾을 수 없습니다."
            state["confidence"] = 0.0
            return state

        # 컨텍스트 구성
        context_parts = []
        total_confidence = 0

        for i, doc in enumerate(retrieved_docs, 1):
            context_part = f"[문서 {i}]\n"
            context_part += f"제목: {doc['title']}\n"
            if doc['metadata']['reg_date']:
                context_part += f"등록일: {doc['metadata']['reg_date']}\n"

            context_part += f"내용: {doc['content']}\n"
            context_part += f"관련도: {doc['similarity_score']:.3f}\n"

            context_parts.append(context_part)
            total_confidence += doc['similarity_score']

        state["context"] = "\n\n".join(context_parts)
        state["confidence"] = total_confidence / len(retrieved_docs)

        return state


    def _generate_streaming_answer_node(self, state: ChatbotState) -> ChatbotState:
        """스트리밍 답변 생성 노드"""
        print(f"[ChatBotGraph] 노드 도착: streaming_answer_generation")
        query = state["query"]
        context = state["context"]
        print(f"[ChatBotGraph] 답변 생성 시작")

        # 향상된 프롬프트 구성
        prompt = f"""당신은 창원대학교의 공지사항을 기반으로 학생들의 질문에 답변하는 AI 어시스턴트입니다.

다음 공지사항들을 참고하여 질문에 대해 정확하고 상세한 답변을 제공해주세요:

{context}

질문: {query}

답변 가이드라인:
1. 공지사항의 정보를 정확히 인용하세요
2. 날짜, 시간, 연락처 등 구체적인 정보를 포함하세요
3. 추가 문의가 필요한 경우 담당 부서나 연락처를 안내하세요
4. 관련 문서를 찾을 수 없다면 솔직히 말씀하세요
5. 한국어로 자연스럽고 친근하게 답변하세요

답변:"""

        try:
            # 스트리밍 LLM API 호출
            full_answer = ""
            for chunk in self.llm_client.call_llm_stream(
                    prompt=prompt,
            ):
                full_answer += chunk
                self.streamingDisplayManager.update(chunk)

            self.streamingDisplayManager.finish()
            print(f"[ChatBotGraph] 답변 생성 완료")

            if full_answer and not full_answer.startswith("LLM 서버"):
                state["answer"] = full_answer
            else:
                state["answer"] = full_answer or "답변을 생성할 수 없습니다."

        except Exception as e:
            state["answer"] = f"답변 생성 중 오류가 발생했습니다: {e}"

        return state


    def _validate_response_node(self, state: ChatbotState) -> ChatbotState:
        """응답 검증 노드"""
        print(f"[ChatBotGraph] 노드 도착: response_validation")
        answer = state["answer"]
        confidence = state["confidence"]
        print(f"[ChatBotGraph] 답변 길이: {len(answer)}, 신뢰도: {confidence:.3f}")

        # 응답 품질 검증
        if len(answer) < 20:
            state["confidence"] *= 0.5

        if any(keyword in answer for keyword in ["오류", "연결", "실패", "수 없습니다"]):
            state["confidence"] *= 0.3

        # 긍정적인 키워드가 있으면 신뢰도 증가
        if any(keyword in answer for keyword in ["안내", "문의", "신청", "참고"]):
            state["confidence"] = min(1.0, state["confidence"] * 1.2)

        # 메타데이터 업데이트
        state["metadata"]["response_length"] = len(answer)
        state["metadata"]["final_confidence"] = state["confidence"]

        return state




