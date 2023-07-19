## 임세훈

### 1. <a href = "https://colab.research.google.com/drive/1RQRLrhGpZboQWftczIMxtgDM9uCm-BwP?usp=sharing"> 코드 </a>
```python
import torch
from torch import nn
input_size = 28*28
output_size = 10
def modelGenerator(array, Sigmoid = False):
  model = nn.Sequential()
  left = 28*28
  right = 10
  for i in array:
    if(i[0] == "L"):
      model.append(nn.Linear(left, i[1]))
      model.append(nn.RReLU()) 
    else:
      exit(-1)
    left = i[1]
  model.append(nn.Linear(left, right));
  if(Sigmoid):
    model.append(nn.Sigmoid())
  return model

model = modelGenerator([("L", 398), ("L", 196), ("L", 98), ("L", 49), ("L", 24), ("L", 12)])
```
- 모델을 직관적으로 나타내고, 수정하기 쉽게 **함수로 작성**
  - Layer 종류/수 같은 **Hyperparameter를 변화시켜가면서 코드를 실행하는 데 유용**
- 각각의 Linear Layer 사이에는 **RReLU Layer 삽입** (과적합 방지)
- Epoch는 30으로 설정

### 2. 학습 결과 
```
Train Accuracy: 90.69%
Test Accuracy: 87.87% 
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 3-1.png?raw=true"/></td>
    </tr>
</table>

