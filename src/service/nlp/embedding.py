from transformers import BertTokenizer, BertModel
import torch

# BERT 모델과 토크나이저를 로드합니다.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 토큰 사용
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # 정규화
    return embeddings
