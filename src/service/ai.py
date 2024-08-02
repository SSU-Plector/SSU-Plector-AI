from flask import abort
from sklearn.metrics.pairwise import cosine_similarity
from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.embedding import get_bert_embeddings
import numpy as np

from src.service.nlp.preprogress import preprocess_text


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

    #Preprocessing
    query = preprocess_text(query)
    short_intro_list = df['short_intro'].apply(preprocess_text).tolist()

    developer_embeddings = get_bert_embeddings(short_intro_list)
    query_embedding = get_bert_embeddings([query])

    if developer_embeddings.shape[0] == 0 or query_embedding.shape[0] == 0:
        abort(400, 'No embeddings found')

    embedding_similarities = cosine_similarity(query_embedding.cpu().numpy(), developer_embeddings.cpu().numpy())

    num_developers = min(5, len(embedding_similarities[0]))
    recommended_indices = np.argsort(-embedding_similarities[0])[:num_developers]

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)
    result_df['similarity'] = embedding_similarities[0, recommended_indices]

    return {
        'developers': [
            {
                'developer_id': row['developer_id'],
                'similarity': row['similarity']
            } for _, row in result_df.iterrows()
        ]
    }
