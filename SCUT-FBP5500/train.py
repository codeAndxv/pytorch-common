import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights
from PIL import Image

from sklearn.metrics import mean_absolute_error, mean_squared_error


class CustomDataset(Dataset):
    def __init__(self, root, filedir, transform=None):
        """
        Args:
            root (string): 图像文件的根目录。
            filedir (string): 带有图像文件名和标签的txt文件路径。
            transform (callable, optional): 转换图像的方法。
        """
        self.root = root
        self.transform = transform

        with open(filedir, 'r') as f:
            lines = f.readlines()

        self.data = []
        for line in lines:
            linesplit = line.split('\n')[0].split(' ')
            addr = linesplit[0]
            target = torch.Tensor([float(linesplit[1])])
            self.data.append((addr, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, target = self.data[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

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


# 训练模型
def train(net, train_loader, optimizer, criterion, scheduler, max_epochs=10, device='cuda'):
    net.train()
    net.to(device)
    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # 每100个batch打印一次loss
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        scheduler.step()
    print('Finished Training')



# 计算皮尔逊相关性
def pearson_correlation_coefficient(y_true, y_pred):
    # 计算协方差矩阵
    cov_matrix = np.cov(y_true, y_pred)
    # 计算标准差
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    # 计算皮尔逊相关性
    pearson_corr = cov_matrix[0, 1] / (std_true * std_pred)
    return pearson_corr

# 计算最大绝对误差和均方根误差
def calculate_errors(y_true, y_pred):
    # 计算最大绝对误差
    mae = mean_absolute_error(y_true, y_pred)
    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# 加载预训练的模型，这里以ResNet为例
net = get_model()

# 设置新的全连接层的参数为可训练
for param in net.fc.parameters():
    param.requires_grad = True

# 设置其他层的参数为可训练
# for param in net.parameters():
#     if param.requires_grad:
#         print(param.shape)  # 打印出可训练参数的形状

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取数据集

root = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/train_test_files/Images'
traindir = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_1/train_1.txt'
train_dataset = CustomDataset(root, traindir, transform=transform)

testdir = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_1/test_1.txt'
test_dataset = CustomDataset(root, testdir, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

# 训练模型
num_epochs = 20000 // len(train_loader)  # 根据迭代次数计算epoch数
for epoch in range(num_epochs):
    net.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 更新学习率
    scheduler.step()

    # 在验证集上评估模型
    net.eval()
    with torch.no_grad():
        valid_loss = 0.0
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = net(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        valid_loss /= len(valid_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

# 保存模型
torch.save(net.state_dict(), './model1.pth')
