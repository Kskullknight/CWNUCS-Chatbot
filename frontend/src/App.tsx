import React from 'react';
import './App.css';
import ChatWidget from './components/ChatWidget';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>창원대학교 공지사항 서비스</h1>
        <p>
          왼쪽 하단의 채팅 아이콘을 클릭하여 공지사항에 대해 문의하세요.
        </p>
      </header>
      
      <main className="App-main">
        <section>
          <h2>서비스 소개</h2>
          <p>
            창원대학교 공지사항 AI 챗봇은 학교 공지사항에 대한 질문에 실시간으로 답변해드립니다.
          </p>
          <ul>
            <li>신속한 정보 검색</li>
            <li>24시간 이용 가능</li>
            <li>정확한 공지사항 안내</li>
          </ul>
        </section>
      </main>

      {/* 채팅 위젯 */}
      <ChatWidget />
    </div>
  );
}

export default App;
