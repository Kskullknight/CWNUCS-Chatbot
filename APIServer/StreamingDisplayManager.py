import sys


class StreamingDisplayManager:
    """스트리밍 출력 관리자"""

    def __init__(self):
        self.current_line = ""
        self.finished = False

    def update(self, new_content: str):
        """새로운 콘텐츠로 현재 라인 업데이트"""
        self.current_line += new_content

        # 실시간 출력 (캐리지 리턴 없이)
        sys.stdout.write(new_content)
        sys.stdout.flush()

    def finish(self):
        """스트리밍 완료"""
        self.finished = True
        print()  # 새 줄 추가