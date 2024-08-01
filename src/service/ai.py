from flask import abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.enum.part import Part
from src.service.database import developer_part_eq
from src.service.nlp.embedding import count_keyword_matches
from src.service.nlp.preprogress import preprocess_text
import numpy as np


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

    # PreProcessing
    query = preprocess_text(query)
    short_intro_list = df['short_intro'].apply(preprocess_text).tolist()

    # Calculate keyword match counts
    match_counts = [count_keyword_matches(query, intro) for intro in short_intro_list]

    # Vectorize the text
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(short_intro_list)
    query_vector = vectorizer.transform([query])

    # Calculate similarity
    embedding_similarities = cosine_similarity(query_vector, vectors).flatten()

    adjusted_similarities = embedding_similarities*0.7 + match_counts*0.3

    num_developers = min(5, len(adjusted_similarities))
    recommended_indices = np.argsort(-adjusted_similarities)[:num_developers]

    # Ensure recommended_indices is 1D
    recommended_indices = np.array(recommended_indices).flatten()

    result_df = df.iloc[recommended_indices].copy()
    result_df['developer_id'] = result_df['developer_id'].astype(int)
    result_df['similarity'] = adjusted_similarities[recommended_indices]

    return {
        'developers': [
            {
                'developer_id': row['developer_id'],
                'similarity': row['similarity']
            } for _, row in result_df.iterrows()
        ]
    }
