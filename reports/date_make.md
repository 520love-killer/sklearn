# 首先先介绍了我们的vit使用的特征处理，然后是常见的机器学习以及深度学习的相关特征处理

| 顺序 | 特征处理方法 | 专业术语                   | 作用                                                         |
| ---- | ------------ | -------------------------- | ------------------------------------------------------------ |
| ①    | 数据归一化   | Normalization              | 将像素值从0~255缩放到0~1之间，减少数据尺度差异，加快模型训练 |
| ②    | 数据标准化   | Standardization（Z-score） | 使数据均值为0、标准差为1，提高训练稳定性和收敛速度           |
| ③    | 随机裁剪     | Random Crop                | 增加样本多样性，提高模型对目标位置变化的鲁棒性               |
| ④    | 随机水平翻转 | Random Horizontal Flip     | 扩充训练数据，提高模型泛化能力                               |
| ⑤    | 标签编码     | Label Encoding             | 将类别名称转换为数字标签，方便模型计算损失函数               |

------

## 详细说明

### 1. 数据归一化（Normalization）

将图像像素：

```text
0~255
```

变成：

```text
0~1
```

例如：

```python
255 → 1.0
128 → 0.5
0 → 0
```

### 作用

- 降低数据尺度差异
- 提高训练效率
- 防止梯度过大

------

### 2. 数据标准化（Standardization）

公式：

[
x'=\frac{x-\mu}{\sigma}
]

其中：

- μ：均值（Mean）
- σ：标准差（Std）

CIFAR-100常用：

```python
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
```

### 作用

- 数据中心化
- 加快模型收敛
- 提高训练稳定性

这是 ViT 必做的预处理。

------

### 3. 随机裁剪（Random Crop）

先在图像周围补边，再随机裁剪回原尺寸。

### 作用

例如：

```text
猫在左边
猫在右边
猫在中间
```

模型都能识别。

提高：

```text
位置鲁棒性
```

减少过拟合。

------

### 4. 随机水平翻转（Random Horizontal Flip）

例如：

```text
汽车朝左
汽车朝右
```

都属于同一类别。

### 作用

- 扩充训练样本
- 提高泛化能力
- 降低过拟合

------

### 5. 标签编码（Label Encoding）

例如：

```text
apple → 0
bus → 1
cat → 2
...
```

### 作用

将类别转换为数字，方便计算：

```python
CrossEntropyLoss
```












这是一个非常重要的认知提升点。

很多同学学深度学习的时候，一上来就接触图像增强、归一化、ViT、CNN，结果误以为：

> 特征工程 = 图像增强

实际上这是错误的。

从机器学习的发展历史来看：

```text
数据预处理
    ↓
特征工程（Feature Engineering）
    ↓
模型训练
    ↓
模型评估
```

其中：

**特征工程（Feature Engineering）才是机器学习最核心的部分。**

甚至在深度学习出现之前：

> 80%的工作量都花在特征工程上。

所以如果你写实验报告或者毕业设计，先介绍通用特征工程，再引出图像领域特征处理，会显得更加专业和系统。

------

# 一、什么是特征工程（Feature Engineering）

定义：

> 利用领域知识对原始数据进行转换、提取、构造和筛选，使数据能够更好地被模型学习。

例如：

原始数据：

| 年龄 |
| ---- |
| 18   |
| 25   |
| 60   |

直接给模型：

```text
18
25
60
```

也可以构造：

| 年龄 | 年龄段 |
| ---- | ------ |
| 18   | 青年   |
| 25   | 青年   |
| 60   | 老年   |

这里：

```text
年龄段
```

就是新构造的特征。

------

# 二、机器学习中的特征工程体系

一般分为：

```text
Feature Engineering
│
├── Data Cleaning
├── Feature Transformation
├── Feature Encoding
├── Feature Scaling
├── Feature Selection
├── Feature Extraction
└── Feature Construction
```

这套体系你可以直接写进实验报告。

------

# 三、数据清洗（Data Cleaning）

## 缺失值处理（Missing Value Handling）

例如：

| 年龄 |
| ---- |
| 20   |
| NaN  |
| 30   |

处理方式：

### Mean Imputation

均值填充

[
x=\frac{\sum x_i}{n}
]

### Median Imputation

中位数填充

### Mode Imputation

众数填充

------

作用：

- 保证数据完整性
- 避免模型报错

------

## 异常值处理（Outlier Detection）

例如：

```text
18
20
22
1000
```

1000明显异常。

常见方法：

### Z-score

[
z=\frac{x-\mu}{\sigma}
]

### IQR

四分位距

[
IQR=Q3-Q1
]

------

作用：

减少噪声影响。

------

# 四、特征编码（Feature Encoding）

机器学习只能处理数字。

因此：

```text
男
女
```

必须转换。

------

## Label Encoding

```text
男 → 0
女 → 1
```

------

## One-Hot Encoding

```text
北京
上海
深圳
```

转换：

```text
北京 1 0 0
上海 0 1 0
深圳 0 0 1
```

作用：

避免类别之间产生大小关系。

------

## Target Encoding

高级方法。

例如：

```text
城市 → 平均房价
```

常用于：

- Kaggle比赛
- CTR预估

------

# 五、特征缩放（Feature Scaling）

这一部分与你的ViT实验关系最大。

------

## Min-Max Normalization

归一化

[
x'=\frac{x-x_{min}}{x_{max}-x_{min}}
]

结果：

```text
0~255
↓
0~1
```

------

作用：

- 加速训练
- 避免数值跨度过大

------

## Standardization

标准化

[
x'=\frac{x-\mu}{\sigma}
]

结果：

```text
均值=0
标准差=1
```

------

作用：

- 梯度更稳定
- 更适合神经网络

------

# 六、特征选择（Feature Selection）

很多变量其实没用。

例如：

预测房价：

```text
房屋面积
楼层
房龄
身份证号
```

身份证号显然无关。

------

## Filter Method

过滤法

### Pearson Correlation

皮尔逊相关系数

[
r=
\frac{cov(X,Y)}
{\sigma_x \sigma_y}
]

------

### Chi-Square Test

卡方检验

分类问题常用。

------

## Wrapper Method

包装法

### Recursive Feature Elimination（RFE）

递归特征消除

不断删除最不重要特征。

------

## Embedded Method

嵌入法

例如：

### LASSO

[
L=L_{MSE}+\lambda|w|
]

自动压缩无用特征。

------

作用：

- 降低过拟合
- 提升训练速度
- 增强可解释性

------

# 七、特征提取（Feature Extraction）

这是很多论文喜欢写的部分。

------

## PCA（主成分分析）

全称：

**Principal Component Analysis**

思想：

```text
100维
↓
10维
```

保留主要信息。

------

作用：

- 降维
- 去噪
- 可视化

------

## LDA

Linear Discriminant Analysis

线性判别分析

目标：

```text
类内距离最小
类间距离最大
```

------

## ICA

Independent Component Analysis

独立成分分析

常用于：

- 信号分离
- EEG分析

------

# 八、特征构造（Feature Construction）

真正体现数据分析能力的地方。

------

例如：

原始特征：

```text
收入
年龄
```

构造：

```text
收入/年龄
```

------

时间数据：

```text
2026-05-29
```

构造：

```text
星期几
月份
季度
是否节假日
```

------

文本数据：

```text
句子
```

构造：

```text
TF-IDF
Word2Vec
BERT Embedding
```

------

# 九、图像领域中的特征工程

现在再回到你的 ViT。

图像其实也遵循同样逻辑：

| 通用机器学习 | 图像领域          |
| ------------ | ----------------- |
| 数据清洗     | 删除损坏图片      |
| 特征缩放     | Normalize         |
| 特征构造     | Data Augmentation |
| 特征提取     | HOG、SIFT         |
| 特征选择     | 通常不用          |
| 降维         | PCA               |

------

# 十、对于你的 ViT-Tony 实验应该怎么写

我建议实验报告按照下面这个逻辑：

```text
1. 数据探索（EDA）
    ├─ 类别统计
    ├─ 样本可视化
    └─ 数据分布分析

2. 数据预处理
    ├─ 数据清洗
    ├─ 特征缩放
    │    └─ Z-score Standardization
    ├─ 数据增强
    │    ├─ Random Crop
    │    ├─ Horizontal Flip
    │    └─ Random Erasing
    └─ 标签编码

3. 特征工程理论分析
    ├─ Feature Scaling
    ├─ Feature Augmentation
    ├─ Feature Extraction
    └─ Feature Construction

4. ViT-Tony模型训练

5. 模型性能对比
```

这样的结构已经接近很多本科毕业设计和课程论文的写法，而不是简单地写一句“进行了归一化和随机翻转处理”。这样老师会看到你不仅会调包训练模型，而且理解了完整的机器学习特征工程体系。

这是一个非常好的问题。

很多学生在写实验报告时，**“特征处理”部分只写了归一化、翻转、裁剪**，实际上老师一看就知道只是照着代码抄的。如果你想让你的 ViT 实验显得更专业，甚至达到本科优秀论文或者研究生课程设计水平，那么你需要知道这些操作背后的专业术语和理论依据。

------

# 一、什么是特征处理（Feature Engineering）

严格来说：

> 特征处理（Feature Processing）是指对原始数据进行变换、提取、构造、筛选和增强，使其更有利于模型学习。

传统机器学习：

```text
原始图片
    ↓
特征提取(HOG、SIFT、LBP)
    ↓
特征选择
    ↓
SVM
```

深度学习：

```text
原始图片
    ↓
预处理
    ↓
CNN/ViT自动学习特征
```

由于 ViT 本身具有自动特征学习能力，所以：

**你的特征处理重点应该放在：**

- Feature Normalization（特征归一化）
- Feature Augmentation（特征增强）
- Feature Standardization（特征标准化）

而不是传统的 HOG、SIFT。

------

# 二、图像领域常见特征处理方法

可以分成五大类。

------

# 第一类：Feature Scaling（特征缩放）

## 1 Min-Max Normalization

公式：

[
x'=\frac{x-x_{min}}{x_{max}-x_{min}}
]

将像素：

```python
0~255
```

映射到：

```python
0~1
```

例如：

```python
255 → 1
128 → 0.5
0 → 0
```

作用：

- 加速收敛
- 防止梯度爆炸

------

## 2 Z-score Standardization

标准化：

[
x'=\frac{x-\mu}{\sigma}
]

其中：

- μ：均值
- σ：标准差

CIFAR-100：

```python
mean = (0.5071,0.4867,0.4408)
std  = (0.2675,0.2565,0.2761)
```

PyTorch：

```python
transforms.Normalize(mean,std)
```

作用：

- 数据中心化
- 提高梯度稳定性
- ViT推荐使用

------

# 第二类：Feature Augmentation（特征增强）

这个是你实验最重要的部分。

------

## Random Crop

随机裁剪

```python
RandomCrop(32,padding=4)
```

原理：

```text
图片
 ↓
四周填充
 ↓
随机截取
```

作用：

提高模型的位置不变性（Translation Invariance）

专业术语：

```text
Spatial Data Augmentation
```

------

## Random Horizontal Flip

随机水平翻转

```python
RandomHorizontalFlip()
```

例如：

```text
猫朝左
猫朝右
```

本质：

增加训练样本多样性。

专业术语：

```text
Geometric Transformation
```

------

## Color Jitter

颜色扰动

```python
ColorJitter()
```

随机改变：

- Brightness
- Contrast
- Saturation
- Hue

专业写法：

```text
Photometric Augmentation
```

------

## Random Rotation

随机旋转

```python
RandomRotation(15)
```

作用：

增强旋转鲁棒性

专业术语：

```text
Rotation Invariance
```

------

# 第三类：Regularization-Based Augmentation

高级增强技术

老师看见这个会觉得你做过文献调研。

------

## Cutout

随机遮挡

例如：

```text
████
██□█
████
```

随机遮住一部分区域。

作用：

防止模型只关注局部区域。

论文：

《Improved Regularization of CNNs with Cutout》

------

## Random Erasing

PyTorch官方实现：

```python
RandomErasing()
```

本质：

Cutout升级版。

专业术语：

```text
Occlusion Augmentation
```

------

## Mixup

2018年经典方法

公式：

[
x=\lambda x_i +(1-\lambda)x_j
]

[
y=\lambda y_i +(1-\lambda)y_j
]

效果：

```text
猫 + 狗
```

生成：

```text
0.7猫 + 0.3狗
```

作用：

增强泛化能力。

------

## CutMix

2020年热门增强

公式：

```text
A图片切一块
↓
贴到B图片
```

标签同步修改。

效果通常优于 Mixup。

------

# 第四类：Feature Extraction（传统特征提取）

这个部分你可以写进实验报告的文献综述。

------

## HOG

Histogram of Oriented Gradients

方向梯度直方图

用于：

- 行人检测

特点：

提取边缘信息。

------

## SIFT

Scale-Invariant Feature Transform

尺度不变特征变换

特点：

- 抗旋转
- 抗缩放

------

## LBP

Local Binary Pattern

局部二值模式

主要用于：

- 人脸识别
- 纹理分析

------

这些在 ViT 中一般不需要。

因为：

```text
ViT自己学习特征
```

------

# 第五类：Feature Selection

传统机器学习很重要。

深度学习很少用。

例如：

## PCA

Principal Component Analysis

主成分分析

作用：

降维

公式：

[
X \rightarrow W^TX
]

保留主要信息。

------

## t-SNE

用于可视化。

实验报告非常好用。

例如：

```text
ViT训练前
类别混杂

ViT训练后
类别分离
```

直接放图。

老师非常喜欢。

------

# 三、对于 ViT-Tony + CIFAR-100 我推荐的方案

如果目标是：

**课程实验 + 写报告 + 准确率较高**

推荐：

```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
    transforms.RandomErasing(p=0.25)
])
```

测试集：

```python
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])
```

------

# 四、实验报告里的专业写法（直接可用）

你完全可以这样写：

> 在特征处理阶段，首先采用 Z-score Standardization 对图像像素进行标准化处理，以减小不同通道间的数据分布差异并提升梯度下降的稳定性。随后采用 Random Crop、Random Horizontal Flip 以及 Color Jitter 等 Spatial Augmentation 和 Photometric Augmentation 技术扩充训练样本空间，提高模型对位置变化及光照变化的鲁棒性。同时引入 Random Erasing 作为 Occlusion-based Regularization Strategy，增强模型对局部遮挡场景的泛化能力，从而有效降低过拟合风险。

这种表述已经接近论文写法了。

------

如果你后面准备做 **ViT-Tony 在 CIFAR-100 上的完整实验（数据探索 → 特征处理 → 模型训练 → 性能对比）**，我还可以告诉你：

**哪些预处理是“必须做的”，哪些是“加分项”，以及哪些对 ViT 的提升最大（按准确率贡献排序）**，这样你的实验设计会更像真正的科研流程。