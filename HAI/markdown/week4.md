## �Ӽ���

### Fashion MNIST �н� ���

#### 1. Feed Forward Network

##### ���
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
##### �ڵ�
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
##### ������
- In/Out Channels �� ����
- Epoch�� $15$�� ����
##### ���
```
Train accuracy: 96.06%
Test accuracy: 91.67%
```

<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-1.png?raw=true"/></td>
    </tr>
</table>


$\rightarrow$ Feed Forward Network ���� �� 4% ���


#### 3. MobileNet(with pretrained weights)

##### �ڵ�
```python
from torchvision.models.mobilenet import mobilenet_v2

# weights �ɼ��� ���� �� �𵨸� �ҷ����ų�
# �����н��� �Ķ���͸� �ҷ����� �� �� ���� ����
model = mobilenet_v2(weights=True)

# Fashion MNIST�� class ������ŭ ����ϵ��� output layer ����
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
_ = model.to(device)
```

##### ���
```
Train accuracy: 98.62%
Test accuracy: 92.27%
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-2.png?raw=true"/></td>
    </tr>
</table>

$\rightarrow$ CNN + MaxPool ���� �� 0.6% ���

#### 4. MobileNet(without pretrained weights)

##### �ڵ�
```python
from torchvision.models.mobilenet import mobilenet_v2

# weights �ɼ��� ���� �� �𵨸� �ҷ����ų�
# �����н��� �Ķ���͸� �ҷ����� �� �� ���� ����
model = mobilenet_v2(weights=False)

# Fashion MNIST�� class ������ŭ ����ϵ��� output layer ����
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
_ = model.to(device)
```

##### ���
```
Train accuracy: 95.53%
Test accuracy: 86.43%
```
<table>
    <tr>
        <td><img src = "https://github.com/mnxcv/Assignments/blob/main/HAI/images/img 4-3.png?raw=true"/></td>
    </tr>
</table>

$\rightarrow$ Pretrained Weights�� ����� ��캸�� �� 6% ����

### ��� �м�
```
Accuracy : 
MobileNet without pretrained weights < Feed Forward < 
CNN + MaxPool < MobileNet with pretrained weights
```

- �����н� ����ġ�� �ҷ��� MobileNet�� ������ ���� ���Ҵ�.
- ����ġ�� �ҷ����� ���� ��� ��Ȯ���� ũ�� ���ϵ��� �� �� �־���.

����� Colab Notebook �ּҴ� <a href = "https://colab.research.google.com/drive/1L3OhbEPT8PmPormXTqWef5zRtRjz-qzc?usp=sharing"> ����</a>���� Ȯ���� �� �ֽ��ϴ�.