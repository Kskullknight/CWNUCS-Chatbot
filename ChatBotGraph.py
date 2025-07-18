from typing import List, Dict, Any, TypedDict, Annotated, Iterator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
import re
import faiss
import numpy as np
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

#나는 임계값을 0.3으로 테스트하긴했는데 그냥 0.5 그대로 둘게
class ChatBotGraph:
    def __init__(self, embedding_model, faiss_index, documents, threshold=0.5):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.documents = documents
        self.threshold = threshold
        self.llm_client = LLMAPIClient()
        self.streamingDisplayManager = StreamingDisplayManager()
        
        self.workflow = self._setup_workflow()
    
    def start_graph(self, query):
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
            result = self.workflow.ainvoke(initial_state)
        except Exception as e:
            result = initial_state
            initial_state['answer'] = f"처리 중 오류가 발생했습니다: {e}"
        
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
        state["answer"] = "죄송합니다. 질문과 관련된 공지사항을 찾을 수 없습니다. 다른 키워드로 다시 질문해 주시거나, 창원대학교 홈페이지를 직접 확인해 주세요."
        state["context"] = "관련 문서를 찾을 수 없습니다."
        state["confidence"] = 0.0
        state["metadata"]["no_documents_reason"] = "검색 결과 없음"
        
        self.streamingDisplayManager.update(state["answer"] + "\n")
        
        return state
    
    
    def _process_query_node(self, state: ChatbotState) -> ChatbotState:
        """쿼리 전처리 노드"""
        query = state.get("query", "")

        # 쿼리 정리 및 확장
        processed_query = self._preprocess_query(query)

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


    def _retrieve_documents_node(self, state: ChatbotState) -> ChatbotState:
        """문서 검색 노드"""
        query = state["query"]

        # try:
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # FAISS 검색
        k = min(10, self.faiss_index.ntotal)  # 인덱스 크기에 따라 조정
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), k)

        # 검색 결과 정리
        retrieved_docs = []
        seen_doc_ids = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):  # 유효하지 않은 인덱스
                continue

            retrieved_document = self.documents[idx]
            doc_id = retrieved_document['doc_id']

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
        
        for dos in retrieved_docs:
            print(dos)

        # except Exception as e:
        #     print(f"리트리버중 에러 발생 {e}")
        #     state["retrieved_docs"] = []

        return state
    
    
    def _prepare_context_node(self, state: ChatbotState) -> ChatbotState:
        """컨텍스트 준비 노드"""
        retrieved_docs = state["retrieved_docs"]

        if not retrieved_docs:
            # TODO: 문서가 없는 경우 바로 종료
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
        query = state["query"]
        context = state["context"]

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

            if full_answer and not full_answer.startswith("LLM 서버"):
                state["answer"] = full_answer
            else:
                state["answer"] = full_answer or "답변을 생성할 수 없습니다."

        except Exception as e:
            state["answer"] = f"답변 생성 중 오류가 발생했습니다: {e}"

        return state


    def _validate_response_node(self, state: ChatbotState) -> ChatbotState:
        """응답 검증 노드"""
        answer = state["answer"]
        confidence = state["confidence"]

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




