## �Ӽ���

### ��� ��
```python
from transformers import pipeline

QA = pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
```
**Question-Answering** task�� �����ϴ� **deepset/roberta-base-squad2** model�� ����Ͽ����ϴ�.

### ��� ����

#### code
```python
news_article = '''
(���� ����)
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

���� ���ÿ� ���� ���ø� �ϳ��� ã�ҽ��ϴ�.

��ü������ ����� �� ����Ǿ� �ִ� Question�� ��� ���� �亯�� ����������
�׷��� ���� ��� �߸��� �亯�� �����ҽ��ϴ�.

����� Colab Notebook �ּҴ� <a href = "https://colab.research.google.com/drive/1LM-qMwvk-V8jroChVS-ldmybR2kLMyDJ?usp=sharing"> ����</a>���� Ȯ���� �� �ֽ��ϴ�.