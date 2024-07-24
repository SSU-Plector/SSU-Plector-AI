from flask import abort
from sklearn.feature_extraction.text import TfidfVectorizer

from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.similarity import max_similarity, cal_similarity

import importlib.util
import sys

# torch 경로를 설정합니다.
torch_path = '/root/.local/lib/python3.11/site-packages/torch'

# torch 모듈을 로드합니다.
spec = importlib.util.spec_from_file_location("torch", f"{torch_path}/__init__.py")
torch = importlib.util.module_from_spec(spec)
sys.modules["torch"] = torch
spec.loader.exec_module(torch)

# 필요한 torch 서브모듈을 임포트합니다.
from torch import tensor, __version__

# 개발자 매칭 함수
def developer_matching(data):
    part = data.get('part')
    query = data.get('request')

    try:
        part_enum = Part(part)
    except ValueError:
        abort(400, '_BAD_REQUEST')

    df = developer_part_eq(part_enum)
    if df.empty:
        return []
    if query is None:
        query = ''

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['short_intro'].tolist())
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_similarities = cal_similarity(tfidf_matrix, query_tfidf)

    recommended_indices = max_similarity(tfidf_similarities, 5)

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)

    return {
        'developers': [{'developer_id': dev_id} for dev_id in result_df['developer_id']]
    }
