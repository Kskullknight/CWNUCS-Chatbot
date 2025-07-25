
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, NoSuchElementException
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
import pandas as pd


def click_next_page(driver, page, timeout=10):
    """
    SAP 웹사이트에서 다음 페이지로 이동하기 위한 페이징 요소 클릭

    Args:
        driver: Selenium WebDriver 인스턴스
        timeout: 요소를 기다리는 최대 시간 (초)
        page:

    Returns:
        bool: 클릭 성공 여부
    """
    try:
        print("다음 페이지 버튼을 찾는 중...")

        # 페이징 요소가 로드될 때까지 대기
        wait = WebDriverWait(driver, timeout)

        # 요소가 존재하고 클릭 가능할 때까지 대기
        next_page_element = wait.until(
            EC.element_to_be_clickable((By.XPATH, f'//*[@id="pagingForm"]/div/ul/li[{page}]/a'))
        )

        print("다음 페이지 버튼을 찾았습니다.")

        # 요소가 화면에 보이도록 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", next_page_element)
        time.sleep(0.5)  # 스크롤 완료 대기

        # 클릭 시도 (여러 방법으로)
        click_success = False

        # 방법 1: 일반 클릭
        try:
            next_page_element.click()
            click_success = True
            print("일반 클릭으로 성공했습니다.")
        except ElementClickInterceptedException:
            print("일반 클릭이 차단됨. 다른 방법을 시도합니다.")

        if click_success:
            # 페이지 로딩 대기
            print("페이지 로딩을 기다리는 중...")
            time.sleep(2)

            # 새 페이지가 로드되었는지 확인 (선택적)
            try:
                # 페이지가 변경되었는지 확인하기 위해 잠시 대기
                WebDriverWait(driver, 5).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
                print("페이지 로딩이 완료되었습니다.")
            except TimeoutException:
                print("페이지 로딩 완료를 확인할 수 없지만 클릭은 성공했습니다.")

            return True
        else:
            print("모든 클릭 방법이 실패했습니다.")
            return False

    except TimeoutException:
        print(f"지정된 시간({timeout}초) 내에 다음 페이지 버튼을 찾을 수 없습니다.")
        return False
    except NoSuchElementException:
        print("다음 페이지 버튼 요소가 존재하지 않습니다.")
        return False
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return False


def extract_data_ids(driver, data_id_set, view_dict, timeout=10):
    """
    지정된 XPath의 tbody에서 tr 요소들의 두 번째 자식 요소에 있는
    a 태그의 data-id 값을 Set에 저장하는 함수

    Args:
        driver: Selenium WebDriver 인스턴스
        data_id_set: 데이터 id를 저장하는 set 객체
        timeout: 요소를 기다리는 최대 시간 (초)

    Returns:
        set: data-id 값들이 저장된 Set 객체
    """

    try:
        # tbody 요소가 로드될 때까지 대기
        wait = WebDriverWait(driver, timeout)
        tbody = wait.until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="nttTable"]/tbody'))
        )

        print("tbody 요소를 찾았습니다.")

        # tbody 아래의 모든 tr 요소들 찾기
        tr_elements = tbody.find_elements(By.TAG_NAME, 'tr')

        print(f"총 {len(tr_elements)}개의 tr 요소를 찾았습니다.")

        for index, tr in enumerate(tr_elements):
            try:
                # tr의 자식 요소들 가져오기 (td 또는 th)
                children = tr.find_elements(By.XPATH, './*')

                # 두 번째 자식 요소가 존재하는지 확인
                if len(children) >= 5:
                    notice = children[0]
                    second_child = children[1]

                    author = children[3].text
                    reg_date = children[3].text
                    view_count = children[4].text

                    # 두 번째 자식 요소에서 a 태그 찾기
                    try:
                        a_element = second_child.find_element(By.TAG_NAME, 'a')

                        # data-id 속성 값 가져오기
                        data_id = a_element.get_attribute('data-id')

                        title = a_element.text

                        if data_id:
                            data_id_set.add(data_id)
                            if len(notice.find_elements(By.XPATH, './*')) > 0:
                                no = True
                            else:
                                no = False
                            view_dict[data_id] = [title, author, reg_date, view_count, no]
                            print(f"TR {index + 1}: data-id = {data_id}")
                        else:
                            print(f"TR {index + 1}: a 태그에 data-id 속성이 없습니다.")

                    except NoSuchElementException:
                        print(f"TR {index + 1}: 두 번째 자식 요소에 a 태그가 없습니다.")

                else:
                    print(f"TR {index + 1}: 자식 요소가 2개 미만입니다. (현재: {len(children)}개)")

            except Exception as e:
                print(f"TR {index + 1} 처리 중 오류: {e}")
                continue

    except TimeoutException:
        print(f"지정된 시간({timeout}초) 내에 tbody 요소를 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

    print(f"\n추출 완료! 총 {len(data_id_set)}개의 고유한 data-id 값을 찾았습니다.")
    print(f"data-id 값들: {list(data_id_set)}")


def html_to_markdown(html):
    soup = BeautifulSoup(html, 'html.parser')

    # 테이블 처리
    for table in soup.find_all('table'):
        table_md = []
        # 헤더
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        if headers:
            table_md.append('| ' + ' | '.join(headers) + ' |')
            table_md.append('|' + '---|' * len(headers))
        # 데이터 행
        for tr in table.find_all('tr'):
            cols = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cols:
                table_md.append('| ' + ' | '.join(cols) + ' |')
        table.replace_with('\n'.join(table_md))

    # 링크 처리
    for a in soup.find_all('a'):
        link_text = a.get_text(strip=True)
        link_url = a.get('href', '')
        if link_url:
            a.replace_with(f'[{link_text}]({link_url})')
        else:
            a.replace_with(link_text)

    # 볼드, 이탤릭 등 단순 변환
    for b in soup.find_all(['b', 'strong']):
        b.string = f"**{b.get_text(strip=True)}**"
    for i in soup.find_all(['i', 'em']):
        i.string = f"*{i.get_text(strip=True)}*"

    # ul/ol/li 목록 처리
    for ul in soup.find_all('ul'):
        items = []
        for li in ul.find_all('li'):
            items.append(f"- {li.get_text(strip=True)}")
        ul.replace_with('\n'.join(items))
    for ol in soup.find_all('ol'):
        items = []
        for idx, li in enumerate(ol.find_all('li'), 1):
            items.append(f"{idx}. {li.get_text(strip=True)}")
        ol.replace_with('\n'.join(items))

    # 줄바꿈 <br> 처리
    for br in soup.find_all('br'):
        br.replace_with('\n')

    # 각 p 태그는 한 줄로 변환
    lines = []
    for child in soup.find_all(['p', 'table']):
        text = child.get_text(separator=' ', strip=True)
        # 테이블은 이미 마크다운으로 변환되어 text에 있음
        if child.name == 'table':
            lines.append(text)
        else:
            if text:
                lines.append(text)
        # p 태그 처리 후 제거(중복 방지)
        child.decompose()

    # p/table 이외 나머지 텍스트(주의: 테이블, p를 모두 처리한 뒤 나머지 텍스트 추출)
    body_text = soup.get_text(separator='\n', strip=True)
    body_text = re.sub(r'\n+', '\n', body_text)
    if body_text:
        lines.append(body_text)

    # 불필요한 연속 개행 제거
    markdown = '\n'.join([line.strip() for line in lines if line.strip()])
    return markdown


def content_to_md(driver, url):
    driver.get(url)

    # 3. XPath로 요소 가져오기
    xpath = '//*[@id="nttViewForm"]/div[2]'
    element = driver.find_element(By.XPATH, xpath)
    html_content = element.get_attribute('outerHTML')

    markdown_content = html_to_markdown(html_content)

    # 5. 결과 출력 (혹은 파일로 저장)
    return markdown_content


def main(pages):
    """
    사용 예시 함수
    """
    # Chrome WebDriver 설정

    chrome_options = Options()
    # 헤드리스 모드 (브라우저 창을 띄우지 않음)
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    # WebDriver 초기화
    driver = webdriver.Chrome(options=chrome_options)
    data_ids_set = set()
    view_dict = {}
    rows = []

    try:
        # 웹페이지 로드 (실제 URL로 변경하세요)
        driver.get("https://www.changwon.ac.kr/ce/na/ntt/selectNttList.do?mi=6627&bbsId=2187")  # 실제 URL로 변경

        # 페이지 로딩을 위한 잠시 대기
        time.sleep(2)

        # data-id 값들 추출
        extract_data_ids(driver, data_ids_set, view_dict)

        for i in range(pages):
            print(i)
            click_next_page(driver, ((i % 10) + 3))

            extract_data_ids(driver, data_ids_set, view_dict)

        print(f"\n=== 최종 결과22 ===")
        print(f"Set 크기: {len(data_ids_set)}")
        print(f"Set 내용: {data_ids_set}")
        print(view_dict)

        for data_id in data_ids_set:
            url = f"https://www.changwon.ac.kr/ce/na/ntt/selectNttInfo.do?mi=6627&nttSn={data_id}"
            content = content_to_md(driver, url)
            title, author, reg_data, view_count, no = view_dict[data_id]
            rows.append(
                {'title': title, 'content': content, 'reg_date': reg_data, 'view_count': view_count, 'author': author,
                 'notice': no})

        df = pd.DataFrame(rows)
        df.to_csv('output.csv', index=False, encoding='utf-8',
                  columns=['title', 'content', 'reg_date', 'view_count', 'notice'])

    except Exception as e:
        print(f"메인 함수 실행 중 오류: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main(10)
