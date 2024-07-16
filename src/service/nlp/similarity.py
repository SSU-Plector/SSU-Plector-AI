from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(matrix, vectorizer, query):
    if query is None:
        query = ''
    query = vectorizer.transform([query])
    similarities = cosine_similarity(matrix, query)
    return similarities


def max_similarity(df, similarities, n=1):
    indices = similarities.argsort(axis=0)[::-1].flatten()
    return [df.iloc[idx] for idx in indices[:n]]
