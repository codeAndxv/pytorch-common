import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights


def get_model():
    # 加载预训练的 ResNet-18 模型，并设置pretrained=True
    resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # 冻结 ResNet-18 的权重
    for param in resnet.parameters():
        param.requires_grad = False

    # 修改最后的全连接层以适应特定任务
    num_features = resnet.fc.in_features
    print(num_features)
    resnet.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    return resnet

get_model()