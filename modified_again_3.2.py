#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
# import SimpleITK as sitk


# In[2]:


fixed_path = glob.glob("/Images/*.jpg")   # 导入所有图片
moving_path = glob.glob("/Image_New_INF/*.jpg")   # 导入所有图片


# # 定义自定义数据集函数

# 关于您的观察：
# 
# 1. **取消旋转**：**transforms.RandomAffine**确实包含旋转作为可能的变换。但由于它是随机的，不是每次都会应用旋转。换句话说，有时可能只应用了平移或缩放，而没有旋转。因此，运行多次可能会得到不同的结果。
# 
# 2. **预览图像的位置下移**：这可能是由于**RandomAffine**应用了平移变换。**translate=(0.1, 0.1)**参数允许图像在水平和垂直方向上随机移动最多10%。因此，如果您观察到图像稍微下移，这可能是由于这种随机平移。
# 
# 如果您希望看到更明显的旋转，可以尝试多次运行数据加载和显示代码，或者调整**RandomAffine**的参数以减少平移和缩放，增加旋转的概率。

# In[3]:


# 再次导入必要的库  
import torchvision.transforms as transforms  
import torch.utils.data as data  
from PIL import Image  
  
# 1. 修改用于预处理的转换  
modified_transform = transforms.Compose([  
    # 将图像转换为张量  
    transforms.ToTensor(), 
    # 添加潜在的数据增强步骤（例如，随机旋转15度，水平和垂直方向上移动0.1像素，并且将像素值在0.9和1.1之间缩放）  
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    # 将图像标准化到[-1, 1]范围内  
    transforms.Normalize(mean=0.5, std=0.5)    
])

  
# 2. 修改自定义数据集  
class ModifiedEyesDataset(data.Dataset):  
    """  
    修改过的眼睛数据集类  
  
    Args:  
        fixed_path (str): 固定图像的路径  
        moving_path (str): 运动图像的路径  
  
    Attributes:  
        fixed_path (str): 固定图像的路径  
        moving_path (str): 运动图像的路径  
    """  
  
    def __init__(self, fixed_path, moving_path):  
        self.fixed_path = fixed_path  
        self.moving_path = moving_path  
  
    def __getitem__(self, index):  
        """  
        获取数据集中的一个样本  
  
        Args:  
            index (int): 样本索引  
  
        Returns:  
            tuple: 包含固定图像、运动图像、固定图像路径和运动图像路径的元组  
        """  
        fixed_path = self.fixed_path[index]  
        moving_path = self.moving_path[index]  
  
        # 加载固定图像，如果图像不是RGB模式，则转换为RGB模式  
        fixed_img = Image.open(fixed_path)  
        if fixed_img.mode != 'RGB':  
            fixed_img = fixed_img.convert('RGB')  
        # 对固定图像进行预处理  
        fixed_img = modified_transform(fixed_img)  
  
        # 加载运动图像，如果图像不是RGB模式，则转换为RGB模式  
        moving_img = Image.open(moving_path)  
        if moving_img.mode != 'RGB':  
            moving_img = moving_img.convert('RGB')  
        # 对运动图像进行预处理  
        moving_img = modified_transform(moving_img)  
  
        return fixed_img, moving_img, fixed_path, moving_path  
  
    def __len__(self):  
        """  
        返回数据集的大小，即固定图像和运动图像中的较小数量  
        """  
        # 返回固定图像和运动图像数量的较小值作为数据集的大小  
        return min(len(self.fixed_path), len(self.moving_path))


# In[4]:


dataset = ModifiedEyesDataset(fixed_path, moving_path)   # 传参的就是所有文件路径的列表
len(dataset)                   # 打印数据集大小


# In[5]:


BTACH_SIZE = 2                               # 批次大小
dataloader = torch.utils.data.DataLoader(
                                       dataset,
                                       batch_size=BTACH_SIZE,
                                       shuffle = False,          # shuffle=True用于打乱数据集，每次都会以不同的顺序返回。
                                       drop_last = True
)
# 如果dataset的大小并不能被batch_size整除，
# 则dataloader中最后一个batch可能比实际的batch_size要小。
# 例如，对于1001个样本，batch_size的大小是10，train_loader的长度len(train_loader)=101，
# 最后一个batch将仅含一个样本。可以通过设置dataloader的drop_last属性为True来避免这种情况。


# In[6]:


len(dataloader)


# # 绘制预处理后批次中的前n张图

# <font size=3>enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，  
# 同时列出数据和数据下标，一般用在 for 循环当中</font>

# In[7]:


fixed_img, moving_img, fixed_filenames, moving_filenames = next(iter(dataloader))   # 返回一个批次的训练数据，BTACH_SIZE大小为多少，就返回几个数据


# In[8]:


fixed_img.shape,moving_img.shape    # 这里是查看情况


# In[ ]:


# type(next(iter(dataloader)))  # 查看一下类型，为list


# In[ ]:


# fixed_filenames,moving_filenames   # 查看加载器是否正确输出


# ## 依次输出

# In[ ]:


# 遍历 fixed_img 中的每个图像
for i, img in enumerate(fixed_img[:10]):
    # 将图像从 PyTorch Tensor 格式还原为 NumPy 数组，并还原通道顺序和归一化操作
    img = (img.permute(1, 2, 0).numpy() + 1) / 2
    
    # 从 fixed_filenames 中获取当前图像的文件名并去除扩展名
    filename = os.path.basename(fixed_filenames[i]).split('.')[0]
    
    # 设置图像的标题
    plt.title(filename, size=20)
    
    # 显示图像
    plt.imshow(img)
    
    # 将图像保存到文件，使用当前图像的文件名加上 "_previewed.png" 作为保存文件名
    plt.savefig(filename + "_previewed.png")
    
    # 显示图像
    plt.show()


# ## 一个画板上排列多个子图

# 可以使用matplotlib的子图（subplots）功能来在一个画板上排列多个子图，使输出更加整齐

# In[ ]:


import os
import matplotlib.pyplot as plt

# 创建一个 2x5 的子图画板，每行显示 5 个图像
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# 遍历 fixed_img 中的每个图像
for i, img in enumerate(fixed_img[:10]):
    # 将图像从 PyTorch Tensor 格式还原为 NumPy 数组，并还原通道顺序和归一化操作
    img = (img.permute(1, 2, 0).numpy() + 1) / 2
    
    # 从 fixed_filenames 中获取当前图像的文件名并去除扩展名
    filename = os.path.basename(fixed_filenames[i]).split('.')[0]
    
    # 获取当前子图的坐标轴对象
    ax = axes[i // 5, i % 5]
    
    # 设置图像的标题
    ax.set_title(filename, size=10)
    
    # 显示图像在当前子图
    ax.imshow(img)
    
    # 将图像保存到文件，使用当前图像的文件名加上 "_previewed.png" 作为保存文件名
    plt.savefig(filename + "_previewed.png")
    
# 调整子图布局，避免重叠
plt.tight_layout()

# 保存整个子图画板为一个文件
plt.savefig("all_previewed_images.png")

# 显示子图画板
plt.show()


# # 创建模型

# ## 自定义库

# In[ ]:


# import sys
# sys.path.insert(0, 'E:/pycharm/voxelmorph-master')

# import voxelmorph.torch.networks as vm_networks
# from voxelmorph.torch.networks import Unet
# from voxelmorph.torch.losses import NCC, MSE, Grad
# from voxelmorph.torch.networks import VxmDense
# print(vm_networks.__file__)


# 使用自定义的库版本，而不是系统路径中的库版本。
# 
# 要在不修改系统路径中的库的情况下导入自定义版本的库，您可以使用以下步骤：
# 
# 1. **修改Python的sys.path**：`sys.path` 是一个列表，Python在其中查找模块。您可以在此列表的开头添加自定义库的路径，以确保Python首先在该位置查找模块。
# 2. **导入模块**：导入您需要的模块。由于您已经修改了 `sys.path`，Python将使用您指定的版本。
# 
# 
# 下述代码首先将您自定义的库路径添加到 `sys.path` 的开头，然后导入模块。最后，它打印出实际导入的模块的路径，以供您确认。
# 
# 请尝试上述步骤，并告诉我结果。

# ## 使用系统库

# In[9]:


# import sys
# sys.path.insert(0, 'E:/pycharm/voxelmorph-master')

import voxelmorph.torch.networks as vm_networks
from voxelmorph.torch.networks import Unet
from voxelmorph.torch.losses import NCC, MSE, Grad
from voxelmorph.torch.networks import VxmDense
print(vm_networks.__file__)


# In[11]:


# 创建VxmDense模型实例
model = VxmDense(inshape=(2912, 2912))    # 用元组传递给inshape参数，元组中有几个值，就决定了调用2d还是3d卷积操作


# # 将模型设置为评估模式
# model.eval()

# # 使用GPU（如果可用）
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# model.to(device)
# fixed_img = fixed_img.to(device)
# moving_img = moving_img.to(device)

# # 前向传播
# warped_img, flow = model(fixed_img, moving_img, registration=True)

# # ...后续代码（例如损失计算和优化）


# # 重写模型

# In[27]:





# In[ ]:


import torch.nn as nn
from voxelmorph.torch.networks import VxmDense, Unet

class ModifiedUnet(Unet):
    def __init__(self, dim, enc_nf, *args, **kwargs):
        super(ModifiedUnet, self).__init__(dim=dim, enc_nf=enc_nf, *args, **kwargs)
        
        # 修改第一个卷积层以接受3个通道的输入
        self.encoder[0][0] = self.conv_block(3, self.enc_nf[0])

class ModifiedVxmDense(VxmDense):
    def __init__(self, inshape, *args, **kwargs):
        super(ModifiedVxmDense, self).__init__(inshape=inshape, *args, **kwargs)
        
        # 使用修改后的Unet模型
        self.unet_model = ModifiedUnet(dim=2, enc_nf=self.enc_nf)

# 创建模型实例
model = ModifiedVxmDense(inshape=(2912, 2912))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



import torch.optim as optim

# 定义图像相似性度量 (例如，均方误差)
mse_loss = nn.MSELoss()

# 定义变形场的正则化 (这是一个占位符，可能需要一个具体的实现)
def regularization_loss(deformation_field):
    return some_regularization_term

# 组合损失
def combined_loss(predicted, target, deformation_field):
    return mse_loss(predicted, target) + regularization_loss(deformation_field)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 打印优化器的信息以供预览
print(optimizer)


# In[ ]:



# 定义训练的 epoch 数量
num_epochs = 10

# 存储每个 epoch 的平均损失
average_losses = []

# 定义训练循环
for epoch in range(num_epochs):
    epoch_losses = []
    for fixed_img, moving_img in dataloader:  # dataloader 是数据加载器
        
        # 前向传播
        deformation_field = model(fixed_img, moving_img)
        
        # 使用变形场将移动图像配准到固定图像
        warped_moving_img = apply_deformation(moving_img, deformation_field)
        
        # 计算损失
        loss = combined_loss(warped_moving_img, fixed_img, deformation_field)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 优化器步骤
        optimizer.step()
        
        # 存储当前的损失
        epoch_losses.append(loss.item())
    
    # 计算并存储这个 epoch 的平均损失
    average_losses.append(sum(epoch_losses) / len(epoch_losses))

# 可视化训练损失
import matplotlib.pyplot as plt
plt.plot(average_losses)
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss Over Epochs')
plt.show()


# In[ ]:



# 设置模型为评估模式
model.eval()

# 用于存储所有配准后的图像和损失的列表
all_warped_images = []
all_losses = []

# 在验证数据上评估模型
for fixed_img, moving_img in validation_dataloader:  # validation_dataloader 是验证数据的加载器
    
    # 使用模型预测变形场
    with torch.no_grad():
        deformation_field = model(fixed_img, moving_img)
    
    # 使用变形场将移动图像配准到固定图像
    warped_moving_img = apply_deformation(moving_img, deformation_field)
    
    # 计算损失
    loss = combined_loss(warped_moving_img, fixed_img, deformation_field)
    
    # 存储结果
    all_warped_images.append(warped_moving_img)
    all_losses.append(loss.item())

# 计算平均损失
average_loss = sum(all_losses) / len(all_losses)

# 输出平均损失
print(f'验证数据的平均损失: {average_loss:.4f}')

# 可视化一些结果
for i in range(3):  # 可视化前三个结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(fixed_img[i].squeeze(), cmap='gray')
    plt.title('固定图像')
    
    plt.subplot(1, 3, 2)
    plt.imshow(moving_img[i].squeeze(), cmap='gray')
    plt.title('移动图像')
    
    plt.subplot(1, 3, 3)
    plt.imshow(warped_moving_img[i].squeeze(), cmap='gray')
    plt.title('配准后的移动图像')
    
    plt.show()

