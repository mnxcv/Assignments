## �Ӽ���

### �ڵ�
```py
import numpy as np
import math
def cal_score(a, b):
    # 0��° ��ū�� �ش��ϴ� �Ӻ��� ���͸� ���
    a = a[0:1]
    b = b[0:1]

    # �� ���� ����ȭ
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    # ����ȭ�� �� ���͸� ���Ͽ� ���絵 ���� ���
    return torch.mm(a_norm, b_norm.transpose(0, 1)).item() *100

def proj_sim(a, b):
  a = a[0:1]
  b = b[0:1]

  diff = torch.matmul(a, b.transpose(0, 1)) / (b.norm(dim = 1) ** 2)[:, None]
  diff = abs(diff[0].item() - 1)
  return 2 / (1 + math.e ** diff)

```

```python
from tqdm import tqdm
def get_sentence_embeddings(sentences):
  # �Է� ������ tokenizer�� ����Ͽ� ��ū �ε��� �迭�� ��ȯ�մϴ�.
  inputs = tokenizer(sentences, padding = True, truncation = True, return_tensors = "pt").to(device)
  # ���� Ȱ���Ͽ� �Ӻ����� ����մϴ�..
  with torch.no_grad():
    embeddings, loss = model(**inputs, return_dict = False) 
  return embeddings

def get_similarity_scores(embeddings, input_text, cal):
  # �Է� �ؽ�Ʈ�� ���� �Ӻ����� ����մϴ�.
  input = []; input.append(input_text);
  input_embedding = get_sentence_embeddings(input)[0] # <- Input text�� ���� Vector

  # �Է� �ؽ�Ʈ�� �Ӻ����� �̸� ���� �Ӻ��� ������ ���絵�� ����մϴ�.
  scores = []
  for emb in embeddings:
    scores.append(cal(emb, input_embedding))

  return scores
```
```python
# �� �κ��� �������� ������!

def main(cal):
  # ���� ��� ������ �Ӻ��� ���� ���
  news_embeddings = get_sentence_embeddings(news_titles)

  input_text = input("�Է� > ")
  # �Էµ� �ؽ�Ʈ�� ���� ��� ����� ������ ���絵 ���
  similarity_scores = get_similarity_scores(news_embeddings, input_text, cal)

  print("��� > �Է°� ������ ���� 5���� ���� ������ ����մϴ�.")
  # ���絵 ������ ���� ���� 5���� index ã��
  top_5_idx = reversed(np.argsort(similarity_scores)[-5:])

  # �� index�� �ش��ϴ� ���� ����� ���絵 ���� ���
  for i, idx in enumerate(top_5_idx):
    print(f"{i+1}. {news_titles[idx]}\t(score: {format(similarity_scores[idx], '.2f')})")
```

- main�Լ��� �Լ��� �μ��� �޵��� �߰��ؼ� ���ο� Calculation Function�� �����ϱ� ������ �����Ͽ����ϴ�.
- <a href = "https://www.linkedin.com/pulse/data-wars-cosine-vs-projection-similarity-ramon-serrallonga">�� ��</a>�� �����ؼ� Projection Similarity�� �߰��� �����Ͽ����ϴ�.

### ����

```
�Է� > ���
��� > �Է°� ������ ���� 5���� ���� ������ ����մϴ�.
1. �����ȭ���׷� ��� ���� ���� ���ǥ	(score: 59.60)
2. �ݵ��Ǽ� �泲 õ�� ������ õ�� ������ �о�	(score: 42.14)
3. ���ٽ�AI �ϳ��������ε� �÷����� ���� üũ�� ����	(score: 41.83)
4. �׷��� ������ ������ �ѱ��� ���� ����� ����	(score: 40.89)
5. �����簡 ������ ������� �𸣳����������� �� ���̵� ����	(score: 38.35)
�Է� > ä������
��� > �Է°� ������ ���� 5���� ���� ������ ����մϴ�.
1. ǻ������ ���Ӱ����� ģȯ�� ��õ... �ݷ������� �Բ��ϴ� �� �޲��� Weekend �ݷ�����	(score: 35.69)
2. ��������� �÷��� ������ ������ ��ö ����� �� ���� ������	(score: 32.36)
3. Ǯ���� �˷���ĭ ���»� ����ī���� ���	(score: 29.07)
4. �ν����̺� �� ���� ���� ��W�ξ�� ��Ÿ ����	(score: 28.78)
5. �ݹ��߰� ������ ����	(score: 28.58)
```

- "���"�� "ä������"�� ����� ���� �ٸ��� ���Խ��ϴ�. ���� ���� "���"�̶�� ��õǾ����� ������ �ܷ�� ���ؼ��� ��Ȯ���� ���� ������ �� �����ϴ�.
- Projection�� Cosine Similarity�� ���ؼ� ����� �ſ� ����ϰ� ���Խ��ϴ�. ��� ����� �ణ ���̴� ������, ��������� ���Ͱ� ���� ������ ������ ��Ȯ���� ���� ������ ���� �����ϱ� ������ ���� �������� ���Դϴ�.

����� Colab Notebook �ּҴ� <a href = "https://colab.research.google.com/drive/11NuyUTwNl4pla8f2dZJDTvHh0q2_VJ62?usp=sharing"> ����</a>���� Ȯ���� �� �ֽ��ϴ�.