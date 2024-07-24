from flask import abort
from sklearn.feature_extraction.text import TfidfVectorizer

from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.embedding import get_bert_embeddings
from src.service.nlp.similarity import max_similarity, cal_similarity


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

    developer_embeddings = get_bert_embeddings(df['short_intro'].tolist())
    query_embedding = get_bert_embeddings([query])
    embedding_similarities = cal_similarity(developer_embeddings, query_embedding)

    hybrid_similarities = 0.5 * tfidf_similarities + 0.5 * embedding_similarities
    recommended_indices = max_similarity(hybrid_similarities, 5)

    # recommended_indices = max_similarity(tfidf_similarities, 5)

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)

    return {
        'developers': [{'developer_id': dev_id} for dev_id in result_df['developer_id']]
    }