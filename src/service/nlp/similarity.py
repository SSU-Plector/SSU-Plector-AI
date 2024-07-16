from sklearn.metrics.pairwise import cosine_similarity


def cal_similarity(matrix, query):
    similarities = cosine_similarity(matrix, query)
    return similarities


def max_similarity(similarities, n=1):
    indices = similarities.flatten().argsort()[::-1]
    return indices[:n]