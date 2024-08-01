import re
from konlpy.tag import Okt
from konlpy.tag import Hannanum

okt = Okt()
hannanum = Hannanum()

def preprocess_text(text):
    # 소문자 변환
    text = text.lower()

    # 특수 문자 제거 (한글과 알파벳, 공백만 허용)
    text = re.sub(r'[^\w\s가-힣]', '', text)

    # 숫자 제거
    text = re.sub(r'\d+', '', text)

    # 특정 패턴 제거
    patterns = [
        r'개발자',  # 단어 제거
        r'<[^>]+>',  # HTML 태그
        r'\b[가-힣]{1,2}\b',  # 1~2글자 한글 단어 (예: 고, 한)
        r'요즘',
        r'최근',
        r'많(이|아요|습니다)?',
        r'좋(아합니다|아해요|겠어|겠어요|겠다|습니다|다)?',
        r'사용(합니다|해요)?',
        r'관심',
        r'이슈',
   ]

    for pattern in patterns:
        text = re.sub(pattern, '', text)

    tokens = hannanum.morphs(text)

    KOREAN_STOP_WORDS = {'가', '이', '은', '는', '이', '가', '의', '과', '에서', '하고', '으로', '에', '에게', '로', '다', '하', '들', '을',
                         '를', '에서', '있','없','있습니다', '없습니다', '합니다', '되다', '그', '저', '저희', '우리', '그들', '그것', '어디','이면','적','인',
                         '언제', '입니다', '하는', '걸', '또한', '저희', '지금', '그저', '너무', '어쩌면','있는','사람','습니다ㄴ','ㅂ니다','스타일','그리고','그러나'}

    # 불용어 제거
    tokens = [word for word in tokens if word not in KOREAN_STOP_WORDS]

    # 특정 접미사 제거 (예: '하고' 제거)
    suffixes_to_remove = ['하고', '한테', '의', '에서', '다', 'ㄴ', '은', '는', '이', '가', '의', '로']
    tokens = [word for word in tokens if word not in suffixes_to_remove]

    # 사전 정의된 단어 목록
    predefined_words = ['클린코드','멀티모듈','마이웨이']

    # 보호된 단어들 복원
    i = 0
    while i < len(tokens):
        # 현재 토큰에서 가능한 합쳐진 단어를 찾기 위한 루프
        j = i + 1
        while j <= len(tokens):
            combined_word = ''.join(tokens[i:j])
            if combined_word in predefined_words:
                tokens = tokens[:i] + [combined_word] + tokens[j:]
                break  # 합쳐졌으므로, i를 다시 설정하고 계속 진행
            j += 1
        i += 1  # 다음 토큰으로 이동


    # 형태소 분석에서 단어 조합
    tokens = [word for word in tokens if len(word) > 1]  # 길이가 1인 단어는 제거

    print(tokens)

    return ' '.join(tokens)

