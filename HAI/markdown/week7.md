## 임세훈

### 코드
```py
import numpy as np
import math
def cal_score(a, b):
    # 0번째 토큰에 해당하는 임베딩 벡터만 사용
    a = a[0:1]
    b = b[0:1]

    # 각 벡터 정규화
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]

    # 정규화된 두 벡터를 곱하여 유사도 점수 계산
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
  # 입력 문장을 tokenizer를 사용하여 토큰 인덱스 배열로 변환합니다.
  inputs = tokenizer(sentences, padding = True, truncation = True, return_tensors = "pt").to(device)
  # 모델을 활용하여 임베딩을 계산합니다..
  with torch.no_grad():
    embeddings, loss = model(**inputs, return_dict = False) 
  return embeddings

def get_similarity_scores(embeddings, input_text, cal):
  # 입력 텍스트에 대한 임베딩을 계산합니다.
  input = []; input.append(input_text);
  input_embedding = get_sentence_embeddings(input)[0] # <- Input text에 대한 Vector

  # 입력 텍스트의 임베딩과 미리 계산된 임베딩 사이의 유사도를 계산합니다.
  scores = []
  for emb in embeddings:
    scores.append(cal(emb, input_embedding))

  return scores
```
```python
# 이 부분은 수정하지 마세요!

def main(cal):
  # 뉴스 기사 제목의 임베딩 벡터 계산
  news_embeddings = get_sentence_embeddings(news_titles)

  input_text = input("입력 > ")
  # 입력된 텍스트와 뉴스 기사 제목들 사이의 유사도 계산
  similarity_scores = get_similarity_scores(news_embeddings, input_text, cal)

  print("출력 > 입력과 유사한 상위 5개의 뉴스 제목을 출력합니다.")
  # 유사도 점수가 가장 높은 5개의 index 찾기
  top_5_idx = reversed(np.argsort(similarity_scores)[-5:])

  # 각 index에 해당하는 뉴스 제목과 유사도 점수 출력
  for i, idx in enumerate(top_5_idx):
    print(f"{i+1}. {news_titles[idx]}\t(score: {format(similarity_scores[idx], '.2f')})")
```

- main함수에 함수를 인수로 받도록 추가해서 새로운 Calculation Function을 적용하기 쉽도록 변경하였습니다.
- <a href = "https://www.linkedin.com/pulse/data-wars-cosine-vs-projection-similarity-ramon-serrallonga">이 글</a>을 참고해서 Projection Similarity를 추가로 구현하였습니다.

### 실행

```
입력 > 비건
출력 > 입력과 유사한 상위 5개의 뉴스 제목을 출력합니다.
1. 현대백화점그룹 비건 시장 본격 출사표	(score: 59.60)
2. 반도건설 충남 천안 유보라 천안 두정역 분양	(score: 42.14)
3. 셀바스AI 하나금융파인드 플랫폼서 셀비 체크업 서비스	(score: 41.83)
4. 그래픽 수학자 허준이 한국계 최초 필즈상 수상	(score: 40.89)
5. 정유사가 무슨죄 시장원리 모르나…베이조스 또 바이든 저격	(score: 38.35)
입력 > 채식주의
출력 > 입력과 유사한 상위 5개의 뉴스 제목을 출력합니다.
1. 퓨리나의 지속가능한 친환경 실천... 반려동물과 함께하는 삶 꿈꾸죠 Weekend 반려동물	(score: 35.69)
2. 가정간편식 플랫폼 구스통 엄선한 제철 식재료 및 음식 선보여	(score: 32.36)
3. 풀무원 알래스칸 명태살 ‘볼카츠’ 출시	(score: 29.07)
4. 인스웨이브 웹 공유 서비스 ‘W셰어링’ 베타 오픈	(score: 28.78)
5. 金배추가 한차에 가득	(score: 28.58)
```

- "비건"과 "채식주의"의 결과가 전혀 다르게 나왔습니다. 뉴스 제목에 "비건"이라고 명시되어있지 않으면 외래어에 대해서는 정확도가 낮게 나오는 것 같습니다.
- Projection과 Cosine Similarity에 대해서 결과가 매우 비슷하게 나왔습니다. 계산 방식이 약간 차이는 있지만, 결과적으로 벡터가 많이 벌어져 있으면 정확도가 낮게 나오는 점은 동일하기 때문에 생긴 현상으로 보입니다.

사용한 Colab Notebook 주소는 <a href = "https://colab.research.google.com/drive/11NuyUTwNl4pla8f2dZJDTvHh0q2_VJ62?usp=sharing"> 여기</a>에서 확인할 수 있습니다.