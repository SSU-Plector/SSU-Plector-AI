import re
from konlpy.tag import Hannanum

hannanum = Hannanum()

def preprocess_text(text):
    # 대문자 변환
    text = text.upper()

    # 형태소 분석
    tokens = hannanum.morphs(text)

    KOREAN_STOP_WORDS = {'하고', '으로', '에게', '에서', '되다', '저희', '우리', '그들', '그것', '어디', '이면',
                         '언제', '입니다', '하는', '또한', '저희', '지금', '그저', '너무', '어쩌면', '있는', '사람', 'ㄴ다', 'ㅂ니다',
                         '그리고', '그러나','에도','으면','원하','겠어','다루','겠다','이제', '또는', '그래서', '더욱', '결국', '게다가',
                         '그러','므로', '즉', '때문', '위해', '처럼', '만큼', '따라서','그렇','지만',
                         '무슨', '원해', '구하','ㅂ니다', '찾고', '습니다','어요','아요','관심','면서','요즘','사용','개발자','이슈','스타일'}

    # 불용어 제거
    tokens = [word for word in tokens if word not in KOREAN_STOP_WORDS]

    # 사전 정의된 단어 목록
    predefined_words = [
        '클린코드', '멀티모듈', '데이터사이언티스트', '프론트엔드', '백엔드', '풀스택', '머신러닝', '마이크로서비스', '데이터베이스',
        '모바일', '웹개발', 'UX디자인', 'UI디자인', '테스트자동화', '데이터분석','대규모트래픽', '서버리스',
        '자바스크립트', '자바', 'C언어', 'R언어','서버관리', '네트워크보안', '사이버보안', '기계학습',
        '자연어처리', '컴퓨터비전', '프로덕트매니저', '프로젝트매니저', '기술지원', '솔루션아키텍트'
    ]

    # 보호된 단어들 복원
    i = 0
    while i < len(tokens):
        j = i + 1
        while j <= len(tokens):
            combined_word = ''.join(tokens[i:j])
            if combined_word in predefined_words:
                tokens = tokens[:i] + [combined_word] + tokens[j:]
                break
            j += 1
        i += 1

    # 길이가 1인 단어 제거
    tokens = [word for word in tokens if len(word) != 1]

    # 특수문자 및 점 제거
    final_text = ' '.join(tokens)
    final_text = re.sub(r'[^\w\s]', '', final_text)

    return final_text

