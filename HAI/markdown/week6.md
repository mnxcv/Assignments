## 임세훈

### 사용 모델
```python
from transformers import pipeline

QA = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
```
**Question-Answering** task를 수행하는 **deepset/roberta-base-squad2** model을 사용하였습니다.

### 사용 예시

#### code
```python
news_article = '''
(내용 생략)
'''

BAD_EXAMPLE = {
    'question': "What is the reason of this record-breaking rainfall?",
    'context': news_article
}
print(f"{BAD_EXAMPLE['question']} -> {QA(BAD_EXAMPLE)['answer']}")

GOOD_EXAMPLE = {
    'question': "How many people have been left without power?",
    'context': news_article
}
print(f"{GOOD_EXAMPLE['question']} -> {QA(GOOD_EXAMPLE)['answer']}")
```

#### output
```
What is the reason of this record-breaking rainfall? -> climate change
How many people have been left without power? -> More than 80,000
```

좋은 예시와 나쁜 예시를 하나씩 찾았습니다.

전체적으로 문장과 잘 연결되어 있는 Question의 경우 옳은 답변을 내놓았지만
그렇지 않은 경우 잘못된 답변을 내놓았습니다.

사용한 Colab Notebook 주소는 <a href = "https://colab.research.google.com/drive/1LM-qMwvk-V8jroChVS-ldmybR2kLMyDJ?usp=sharing"> 여기</a>에서 확인할 수 있습니다.