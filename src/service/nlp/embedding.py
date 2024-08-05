import numpy as np
from torch import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


# 한국어 BERT 모델과 토크나이저를 로드합니다.
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = BertModel.from_pretrained('klue/bert-base')

def get_bert_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # 정규화
    return embeddings

def count_keyword_matches(query, intro):
    query_tokens = set(query.split())
    intro_tokens = set(intro.split())
    return len(query_tokens.intersection(intro_tokens))
