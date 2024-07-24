import os
import sys

pytorch_path = os.getenv('PYTORCHPATH')

if pytorch_path:
    if pytorch_path not in sys.path:
        sys.path.append(pytorch_path)

# 이제 torch 모듈을 임포트
import torch
print(f"torch 패키지 버전: {torch.__version__}")

from transformers import BertTokenizer, BertModel

# BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def get_bert_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
