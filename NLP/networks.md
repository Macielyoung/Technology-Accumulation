## 神经网络模型

### 1. CNN系列

CNN模型有三个基本思想，局部感受野(local receptive fields)、权值共享(shared weights)和池化(pooling)。

（1）ResNet模型

假如神经网络的输入是x，期望输出是H(x)，如果我们把输入直接传到输出作为初始结果，那么我们需要学习的目标就是F(x)=H(x)-x。ResNet相当于将学习目标改变了，不再是学习一个完整的H(x)，而是学习输出和输入的差别，即残差F(x)。

ResNet模型中残差连接存在两种连接方式，即下图中的"实线"和"虚线"两种。

![connection](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/connection.png)

实线连接的两边都是`3*3*64`的特征图，他们通道数一致，所以采用计算方式：y=F(x)+x。

虚线连接的两边分别是`3*3*64`和`3*3*128`两种特征图，由于他们通道数不同，因此需要进行变换通道操作，即增加一个卷积操作，它采用的计算方式：y=F(x)+Wx，W是卷积操作。



ResNet论文中提出两种残差学习模块，分为两层和三层这两种学习模块，如下图所示。

![shortcut](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/shortcut.png)

两层结构主要用于ResNet18和ResNet34，而三层结构则用于更深层的网络结构ResNet50，ResNet101和ResNet152。三层结构也称为“bottleneck design”，目的就是为了降低参数的数目。如右图所示。三层结构中的中间3x3的卷积层首先在一个降维1x1卷积层下减少了计算，然后在另一个1x1的卷积层下做了还原，既保持了精度又减少了计算量。第一个1x1的卷积把256维channel降到64维，然后在最后通过1x1卷积恢复，整体上用的参数数目：1x1x256x64 + 3x3x64x64 + 1x1x64x256 = 69632，而不使用bottleneck的话就是两个3x3x256的卷积，参数数目: 3x3x256x256x2 = 1179648，差了16.94倍。

```python
Pytorch中二维卷积操作：
torch.nn.Conv2d(in_channels，out_channels，kernel_size，stride=1，padding=0，dilation=1，groups=1，bias=True)

in_channels：输入维度
out_channels：输出维度
kernel_size：卷积核大小
stride：步长大小
padding：补0
dilation：kernel间距

卷积计算：
d_{out}=(d_{in} - kennel_size + 2 * padding) / stride + 1

示例：
import torch.nn as nn
a = torch.randn(1, 3, 224, 224)
c1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
b = c1(a)
b.shape: torch.Size([1, 64, 112, 112])
```



### 2. RNN系列



### 3. Transformer系列

![transformer](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/transformer.jpeg)

（1）Transformer

Transformer本身还是一个Encoder-Decoder模型，可以认为是一个Seq2Seq with attention的结构。

Encoder端由N个相同大模块堆叠而成。每个大模块由两个子模块组成，分别为多头self-attention模块和前馈神经网络模块。

Decoder端同样由N个相同大模块组成，其中每个大模块由三个子模块组成，分别为多头self-attention模块，多头Encoder-Decoder Attention模块和前馈神经网络结构。Decoder端训练和测试时接收的输入是不同的，训练时输入为上次输入加上输入序列后移一位的Ground Truth（比如一个新单词的词向量或者一段新音频特征），其中时间步为1时是一个特殊的token，用于对应任务设置的特定输入。**实际训练过程中一次性将目标序列的embedding都输入到第一个模块中，然后在多头attention模块中对输入序列进行mask**。测试时是挨个生成对应位置的输出。

Transformer中使用的attention公式是：

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

可描述为**将query和key-value键值对的一组集合映射到输出**。多头self-attention模块，则是将Q、K、V通过参数矩阵映射后（给Q、K、V分别接一个全连接层），然后再做self-attention，最后再将所有结果拼接起来。

对应公式为：

$$MultiHead(Q,K,V)=Concat(head_1,head_2,...,head_h)W^O$$

$$where ~head_i=Attention(QW_i^Q,K_i^K,V_i^V)$$

其中$W_i^Q,W_i^K,W_i^V \in \mathbb{R}^{d_{model}\times d_k}$，$W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

