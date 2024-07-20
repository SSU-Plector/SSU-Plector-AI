from flask import abort
from sentence_transformers import SentenceTransformer, util
from src.enum.part import Part
from src.service.database import developer_part_eq

# SentenceTransformer 모델 로드
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


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

    query_embedding = model.encode(query, convert_to_tensor=True) # 쿼리 문장의 임베딩 생성

    short_intros = df['short_intro'].tolist()
    embeddings = model.encode(short_intros, convert_to_tensor=True) # 데이터프레임의 각 문장에 대한 임베딩 생성

    similarities = util.cos_sim(query_embedding, embeddings)[0] # 문장 간의 유사도 계산

    top_k = min(5, len(similarities))
    top_results = similarities.topk(k=top_k, largest=True) # 유사도 높은 상위 5개 선택

    recommended_indices = top_results.indices.cpu().numpy()

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)

    return {
        'developers': [{'developer_id': dev_id} for dev_id in result_df['developer_id']]
    }
