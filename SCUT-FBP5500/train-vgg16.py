import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, VGG16_Weights
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
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    # 冻结 ResNet-18 的权重
    for param in vgg.parameters():
        param.requires_grad = False

    # 修改最后的全连接层以适应特定任务
    num_features = vgg.classifier[-1].in_features
    vgg.classifier[-1] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    return vgg


# 计算皮尔逊相关性
def pearson_correlation_coefficient(x, y):
    # 计算均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # 计算协方差
    cov_xy = np.mean((x - x_mean) * (y - y_mean))

    # 计算标准差
    std_x = np.std(x)
    std_y = np.std(y)

    # 计算皮尔逊相关系数
    pearson_coefficient = cov_xy / (std_x * std_y)

    return pearson_coefficient


# 计算最大绝对误差和均方根误差
def calculate_errors(y_true, y_pred):
    # 计算最大绝对误差
    mae = mean_absolute_error(y_true, y_pred)
    # 计算均方根误差
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

# 加载预训练的模型，这里以ResNet为例
net = get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 读取数据集

root = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/Images'
traindir = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_1/train_1.txt'
train_dataset = CustomDataset(root, traindir, transform=transform)

testdir = './SCUT-FBP5500_v2.1/SCUT-FBP5500_v2/train_test_files/5_folders_cross_validations_files/cross_validation_1/test_1.txt'
test_dataset = CustomDataset(root, testdir, transform=transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 根据迭代次数计算epoch数
max_epochs = 40

# 训练模型
def train(net, train_loader, test_loader, optimizer, criterion, scheduler, max_epochs, device):

    net.to(device)
    best_mae = float('inf')  # 初始最佳 MAE 为正无穷
    best_model_state_dict = None  # 最佳模型参数的状态字典
    for epoch in range(max_epochs):
        net.train()
        for images, labels in train_loader:
            # 将数据移动到 GPU 上
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

        # 在测试集上评估模型
        net.eval()
        with torch.no_grad():
            predictions = []
            true_labels = []
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # loss = criterion(outputs, labels)
                predictions.extend(outputs.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # 计算皮尔逊相关系数
        pc = pearson_correlation_coefficient(true_labels, predictions)
        # 计算 MAE
        mae = mean_absolute_error(true_labels, predictions)
        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(true_labels, predictions))

        print(f'Epoch [{epoch + 1}/{max_epochs}], Pearson Correlation Coefficient: {pc:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}')

        # 如果当前 MAE 比历史最佳 MAE 更低，则更新最佳 MAE 和最佳模型参数
        if mae < best_mae:
            best_mae = mae
            best_model_state_dict = net.state_dict()
            print(f'保存第[{epoch + 1}/{max_epochs}] 模型')

        # 保存具有最低 MAE 的模型参数
        torch.save(best_model_state_dict, './best_model.pth')

train(net, train_loader, test_loader,optimizer, criterion, scheduler, max_epochs, device)