## �Ӽ���

### 1. <a href = "https://colab.research.google.com/drive/1RQRLrhGpZboQWftczIMxtgDM9uCm-BwP?usp=sharing"> �ڵ� </a>
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
- ���� ���������� ��Ÿ����, �����ϱ� ���� **�Լ��� �ۼ�**
  - Layer ����/�� ���� **Hyperparameter�� ��ȭ���Ѱ��鼭 �ڵ带 �����ϴ� �� ����**
- ������ Linear Layer ���̿��� **RReLU Layer ����** (������ ����)
- Epoch�� 30���� ����

### 2. �н� ��� 
```
Train Accuracy: 90.69%
Test Accuracy: 87.87% 
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 3-1.png?raw=true"/></td>
    </tr>
</table>

