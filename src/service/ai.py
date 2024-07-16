from flask import abort
from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.feature_engineering import tfidf_feature_engineering
from src.service.nlp.similarity import calculate_similarity, max_similarity


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

    tfidf_matrix, vectorizer = tfidf_feature_engineering(df, 'short_intro')
    similarities = calculate_similarity(tfidf_matrix, vectorizer, query)
    recommended_developers = max_similarity(df, similarities, 5)

    for dev in recommended_developers:
        dev['developer_id'] = int(dev['developer_id'])

    return [
        {'developer_id': dev['developer_id']}
        for dev in recommended_developers
    ]