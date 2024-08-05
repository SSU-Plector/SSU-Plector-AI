from src.service.developer_matching.helpers import get_part_enum, get_developer_data, calculate_similarities, \
    get_top_recommendations
from src.service.nlp.preprogress import preprocess_text


def developer_matching(data):
    part = data.get('part')
    query = data.get('request')

    part_enum = get_part_enum(part)
    df = get_developer_data(part_enum)

    query = preprocess_text(query)
    short_intro_list = df['short_intro'].apply(preprocess_text).tolist()

    similarities = calculate_similarities(query, short_intro_list)

    return get_top_recommendations(df, similarities)
