from sklearn.feature_extraction.text import TfidfVectorizer


# TF-IDF를 사용한 피처 엔지니어링
def tfidf_feature_engineering(df, column):

    df[column] = df[column].fillna('')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[column])
    return tfidf_matrix, vectorizer
