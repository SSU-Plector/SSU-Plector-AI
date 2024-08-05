from flask import abort
from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.embedding import count_keyword_matches
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_part_enum(part):
    try:
        return Part(part)
    except ValueError:
        abort(400, '_BAD_REQUEST')


def get_developer_data(part_enum):
    df = developer_part_eq(part_enum)
    if df.empty:
        return []
    return df


def calculate_similarities(query, short_intro_list):
    match_counts = [count_keyword_matches(query, intro) for intro in short_intro_list]
    match_counts = np.array(match_counts)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(short_intro_list)
    query_vector = vectorizer.transform([query])

    embedding_similarities = cosine_similarity(query_vector, vectors).flatten()
    adjusted_similarities = embedding_similarities * 0.7 + match_counts * 0.3
    return np.clip(adjusted_similarities, a_min=None, a_max=1.0)


def get_top_recommendations(df, similarities, top_n=5):
    num_developers = min(top_n, len(similarities))
    recommended_indices = np.argsort(-similarities)[:num_developers]

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)
    result_df['similarity'] = similarities[recommended_indices]

    return {
        'developers': [
            {
                'developer_id': row['developer_id'],
                'similarity': row['similarity']
            } for _, row in result_df.iterrows()
        ]
    }
