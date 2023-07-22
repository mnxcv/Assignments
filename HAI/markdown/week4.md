## 임세훈

### Fashion MNIST 학습 결과

#### 1. Feed Forward Network

##### 결과
```
Train accuracy: 90.69%
Test accuracy: 87.87%
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 3-1.png?raw=true"/></td>
    </tr>
</table>


#### 2. CNN + MaxPool
##### 코드
```python
model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding="valid"),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding="valid"),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(in_features=4*4*256, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=10)
)
```
##### 변경점
- In/Out Channels 값 변경
- Epoch값 $15$로 변경
##### 결과
```
Train accuracy: 96.06%
Test accuracy: 91.67%
```

<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-1.png?raw=true"/></td>
    </tr>
</table>


$\rightarrow$ Feed Forward Network 보다 약 4% 향상


#### 3. MobileNet(with pretrained weights)

##### 코드
```python
from torchvision.models.mobilenet import mobilenet_v2

# weights 옵션을 통해 빈 모델만 불러오거나
# 사전학습된 파라미터를 불러오는 것 중 선택 가능
model = mobilenet_v2(weights=True)

# Fashion MNIST의 class 개수만큼 출력하도록 output layer 변형
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
_ = model.to(device)
```

##### 결과
```
Train accuracy: 98.62%
Test accuracy: 92.27%
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-2.png?raw=true"/></td>
    </tr>
</table>

$\rightarrow$ CNN + MaxPool 보다 약 0.6% 향상

#### 4. MobileNet(without pretrained weights)

##### 코드
```python
from torchvision.models.mobilenet import mobilenet_v2

# weights 옵션을 통해 빈 모델만 불러오거나
# 사전학습된 파라미터를 불러오는 것 중 선택 가능
model = mobilenet_v2(weights=False)

# Fashion MNIST의 class 개수만큼 출력하도록 output layer 변형
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
_ = model.to(device)
```

##### 결과
```
Train accuracy: 95.53%
Test accuracy: 86.43%
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-3.png?raw=true"/></td>
    </tr>
</table>

$\rightarrow$ Pretrained Weights를 사용한 경우보다 약 6% 저하

### 결과 분석
```
Accuracy : 
MobileNet without pretrained weights < Feed Forward < 
CNN + MaxPool < MobileNet with pretrained weights
```

- 사전학습 가중치를 불러온 MobileNet의 성능이 가장 좋았다.
- 가중치를 불러오지 않은 경우 정확도가 크게 저하됨을 알 수 있었다.

사용한 Colab Notebook 주소는 <a href = "https://colab.research.google.com/drive/1L3OhbEPT8PmPormXTqWef5zRtRjz-qzc?usp=sharing"> 여기</a>에서 확인할 수 있습니다.