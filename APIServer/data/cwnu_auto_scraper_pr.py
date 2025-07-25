from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import json
import time
import sys
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

# 전역 락 (JSON 파일 쓰기 동기화용)
file_lock = Lock()
progress_lock = Lock()
progress_data = {'completed': 0, 'total': 0}

def html_to_markdown(html):
    soup = BeautifulSoup(html, 'html.parser')
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

def setup_driver():
    """Chrome 드라이버 설정 (스레드별로 독립적인 드라이버 생성)"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    # 스레드 안전성을 위한 추가 옵션
    chrome_options.add_argument('--disable-features=VizDisplayCompositor')
    
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def get_notice_list(driver, page_num=1):
    """특정 페이지의 게시글 목록 가져오기"""
    list_url = f'https://www.changwon.ac.kr/ce/na/ntt/selectNttList.do?mi=6627&bbsId=2187&currPage={page_num}'
    driver.get(list_url)
    time.sleep(2)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    notices = []
    
    tbody = soup.find('tbody')
    if tbody:
        rows = tbody.find_all('tr')
        for row in rows:
            tds = row.find_all('td')
            if len(tds) >= 6:
                num_text = tds[0].text.strip()
                # 일반 게시글만 (공지 제외)
                if num_text.isdigit():
                    link = tds[1].find('a')
                    if link:
                        ntt_sn = link.get('data-id')
                        if ntt_sn:
                            notice_info = {
                                'number': num_text,
                                'ntt_sn': ntt_sn,
                                'title': link.text.strip(),
                                'author': tds[2].text.strip(),
                                'date': tds[3].text.strip(),
                                'views': tds[4].text.strip()
                            }
                            notices.append(notice_info)
    
    return notices

def scrape_notice_detail(driver, ntt_sn):
    """개별 게시글 상세 정보 크롤링"""
    detail_url = f'https://www.changwon.ac.kr/ce/na/ntt/selectNttInfo.do?mi=6627&nttSn={ntt_sn}'
    driver.get(detail_url)
    time.sleep(1)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    notice_data = {
        'url': detail_url,
        'ntt_sn': ntt_sn
    }
    
    # BD_table 클래스 내부의 테이블에서 정보 추출
    bd_table = soup.find('div', class_='BD_table')
    if bd_table:
        table = bd_table.find('table')
        if table:
            rows = table.find_all('tr')
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                
                if th and td:
                    label = th.text.strip()
                    value = td.text.strip()
                    
                    if th.get('class') and 'title' in th.get('class'):
                        notice_data['title'] = value
                    elif '작성자' in label:
                        notice_data['author'] = value
                    elif '작성일' in label or '등록일' in label:
                        notice_data['date'] = value
                    elif '조회' in label:
                        notice_data['views'] = value
                    elif '첨부파일' in label:
                        attachments = []
                        file_links = td.find_all('a')
                        for link in file_links:
                            file_name = link.text.strip()
                            if file_name:
                                attachments.append({
                                    'name': file_name,
                                    'onclick': link.get('onclick', '')
                                })
                        notice_data['attachments'] = attachments
    
    # 본문 내용 추출
    content_div = soup.find('div', class_='ntt_cn')
    if content_div:
        outer_html = str(content_div).replace('\n', '').replace('\r', '')
        markdown_content = html_to_markdown(outer_html)
        notice_data['content'] = markdown_content
        
        # 이미지 추출
        images = []
        img_tags = content_div.find_all('img')
        for img in img_tags:
            img_data = {
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            }
            if img_data['src']:
                if img_data['src'].startswith('/'):
                    img_data['src'] = 'https://www.changwon.ac.kr' + img_data['src']
                images.append(img_data)
        
        if images:
            notice_data['images'] = images
        
        # 링크 추출
        links = []
        link_tags = content_div.find_all('a')
        for link in link_tags:
            link_data = {
                'href': link.get('href', ''),
                'text': link.text.strip()
            }
            if link_data['href']:
                links.append(link_data)
        
        if links:
            notice_data['links'] = links
    
    # 콘텐츠 타입 결정
    has_content = bool(notice_data.get('content'))
    has_images = bool(notice_data.get('images'))
    
    if has_images and not has_content:
        content_type = 'image_only'
    elif has_content and has_images:
        content_type = 'text_and_image'
    elif has_content:
        content_type = 'text_only'
    else:
        content_type = 'empty'
    
    notice_data['content_type'] = content_type
    
    return notice_data

def scrape_page(page_num, pages_to_scrape):
    """특정 페이지의 모든 게시글 크롤링 (스레드에서 실행)"""
    thread_id = threading.current_thread().name
    driver = setup_driver()
    page_notices = []
    
    try:
        print(f"[{thread_id}] 페이지 {page_num}/{pages_to_scrape} 시작...")
        notices = get_notice_list(driver, page_num)
        
        for notice in notices:
            try:
                detail_data = scrape_notice_detail(driver, notice['ntt_sn'])
                # 목록에서 가져온 정보와 병합
                detail_data['number'] = notice['number']
                if 'title' not in detail_data:
                    detail_data['title'] = notice['title']
                if 'author' not in detail_data:
                    detail_data['author'] = notice['author']
                if 'date' not in detail_data:
                    detail_data['date'] = notice['date']
                if 'views' not in detail_data:
                    detail_data['views'] = notice['views']
                
                page_notices.append(detail_data)
                
                # 진행 상황 업데이트
                with progress_lock:
                    progress_data['completed'] += 1
                    if progress_data['completed'] % 10 == 0:
                        print(f"진행 상황: {progress_data['completed']}/{progress_data['total']} 게시글 완료")
                
                time.sleep(0.05)  # 과도한 요청 방지 (병렬이므로 더 짧게)
                
            except Exception as e:
                print(f"[{thread_id}] 게시글 {notice['number']} 크롤링 오류: {str(e)}")
                continue
        
        print(f"[{thread_id}] 페이지 {page_num} 완료 ({len(page_notices)}개 게시글)")
        return page_notices
        
    except Exception as e:
        print(f"[{thread_id}] 페이지 {page_num} 크롤링 오류: {str(e)}")
        return []
        
    finally:
        driver.quit()

def save_to_csv(notices, filename='output.csv', detailed_filename='output_detailed.csv'):
    """게시글 데이터를 CSV 파일로 저장 (기존 형식 + 상세 정보)"""
    if not notices:
        print("저장할 데이터가 없습니다.")
        return
    
    # 1. 기존 형식 CSV (RAG 시스템용)
    csv_data = []
    for notice in notices:
        # content 필드 정리 (줄바꿈을 공백으로 변환)
        content = notice.get('content', '')
        if content:
            content = ' '.join(content.split())  # 연속된 공백 제거
        
        row = {
            'title': notice.get('title', ''),
            'content': content,
            'reg_date': notice.get('date', ''),
            'view_count': notice.get('views', ''),
            'notice': notice.get('number', '')
        }
        csv_data.append(row)
    
    # 기존 형식 CSV 저장
    df = pd.DataFrame(csv_data)
    df = df[['title', 'content', 'reg_date', 'view_count', 'notice']]
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 (기존 형식): {filename}")
    
    # 2. 상세 정보 포함 CSV (JSON과 동일한 내용)
    detailed_data = []
    for notice in notices:
        # 첨부파일 정보 정리
        attachments = notice.get('attachments', [])
        attachment_names = [att['name'] for att in attachments]
        attachment_count = len(attachments)
        
        # 이미지 정보 정리
        images = notice.get('images', [])
        image_urls = [img['src'] for img in images]
        image_count = len(images)
        
        # 링크 정보 정리
        links = notice.get('links', [])
        link_info = [f"{link['text']} ({link['href']})" for link in links]
        link_count = len(links)
        
        detailed_row = {
            'number': notice.get('number', ''),
            'title': notice.get('title', ''),
            'author': notice.get('author', ''),
            'date': notice.get('date', ''),
            'views': notice.get('views', ''),
            'content': notice.get('content', ''),  # 마크다운 형식
            'content_type': notice.get('content_type', ''),
            'url': notice.get('url', ''),
            'ntt_sn': notice.get('ntt_sn', ''),
            'attachment_count': attachment_count,
            'attachment_names': ' | '.join(attachment_names) if attachment_names else '',
            'image_count': image_count,
            'image_urls': ' | '.join(image_urls) if image_urls else '',
            'link_count': link_count,
            'links': ' | '.join(link_info) if link_info else ''
        }
        detailed_data.append(detailed_row)
    
    # 상세 정보 CSV 저장
    df_detailed = pd.DataFrame(detailed_data)
    df_detailed.to_csv(detailed_filename, index=False, encoding='utf-8-sig')
    print(f"CSV 파일 저장 (상세 정보): {detailed_filename}")

def get_total_pages(driver):
    """전체 페이지 수 가져오기"""
    list_url = 'https://www.changwon.ac.kr/ce/na/ntt/selectNttList.do?mi=6627&bbsId=2187'
    driver.get(list_url)
    time.sleep(3)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # goPaging 함수를 포함하는 모든 링크에서 최대 페이지 번호 찾기
    all_links = soup.find_all('a', onclick=re.compile('goPaging'))
    max_page = 1
    
    for link in all_links:
        onclick = link.get('onclick', '')
        match = re.search(r'goPaging\((\d+)\)', onclick)
        if match:
            page_num = int(match.group(1))
            if page_num > max_page:
                max_page = page_num
    
    # 전체 게시글 수에서 계산하는 방법도 시도
    page_text = soup.get_text()
    total_pattern = re.search(r'총\s*([0-9,]+)\s*건', page_text)
    if total_pattern:
        total_posts = int(total_pattern.group(1).replace(',', ''))
        calculated_pages = (total_posts + 14) // 15  # 페이지당 15개
        max_page = max(max_page, calculated_pages)
    
    print(f"감지된 전체 페이지 수: {max_page}")
    return max_page

def main(pages=1, max_workers=5):
    """메인 실행 함수 (병렬 처리)"""
    # 초기 드라이버로 전체 페이지 수 확인
    initial_driver = setup_driver()
    
    try:
        # 전체 페이지 수 확인
        total_pages = get_total_pages(initial_driver)
        print(f"전체 페이지 수: {total_pages}")
        
        # 크롤링할 페이지 수 결정
        pages_to_scrape = min(pages, total_pages)
        print(f"\n{pages_to_scrape}페이지를 {max_workers}개의 스레드로 병렬 크롤링합니다...")
        
        # 예상 게시글 수 설정 (진행률 표시용)
        progress_data['total'] = pages_to_scrape * 15
        
        # ThreadPoolExecutor를 사용한 병렬 처리
        all_notices = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 각 페이지에 대한 작업 제출
            future_to_page = {
                executor.submit(scrape_page, page, pages_to_scrape): page 
                for page in range(1, pages_to_scrape + 1)
            }
            
            # 완료된 작업에서 결과 수집
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_notices = future.result()
                    all_notices.extend(page_notices)
                except Exception as e:
                    print(f"페이지 {page_num} 처리 중 오류: {str(e)}")
        
        # 번호순으로 정렬 (최신순)
        all_notices.sort(key=lambda x: int(x.get('number', '0')), reverse=True)
        
        # 결과 저장 (JSON과 CSV 모두)
        json_file = 'cwnu_all_notices_parallel.json'
        csv_file = 'output.csv'
        
        # JSON 저장
        with file_lock:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_notices, f, ensure_ascii=False, indent=2)
        
        # CSV 저장
        save_to_csv(all_notices, csv_file)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n✅ 병렬 크롤링 완료!")
        print(f"총 {len(all_notices)}개의 게시글을 수집했습니다.")
        print(f"소요 시간: {elapsed_time:.1f}초")
        print(f"JSON 파일 저장: {json_file}")
        
        # 통계 출력
        print("\n📊 수집된 게시글 통계:")
        content_types = {}
        for notice in all_notices:
            ct = notice.get('content_type', 'unknown')
            content_types[ct] = content_types.get(ct, 0) + 1
        
        for ct, count in content_types.items():
            print(f"  - {ct}: {count}개")
        
        # 첨부파일이 있는 게시글 수
        with_attachments = sum(1 for n in all_notices if n.get('attachments'))
        print(f"  - 첨부파일 포함: {with_attachments}개")
        
    except Exception as e:
        print(f"크롤링 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        initial_driver.quit()

if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == 'all':
            pages = 9999  # 충분히 큰 수
        else:
            try:
                pages = int(arg)
            except:
                pages = 1
    else:
        pages = 1
    
    # 스레드 수 (선택적)
    max_workers = 5  # 기본값
    if len(sys.argv) > 2:
        try:
            max_workers = int(sys.argv[2])
            max_workers = min(max_workers, 10)  # 최대 10개로 제한
        except:
            pass
    
    main(pages, max_workers)