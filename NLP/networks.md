## 神经网络模型

### 一、CNN系列

CNN模型有三个基本思想，局部感受野(local receptive fields)、权值共享(shared weights)和池化(pooling)。

#### 1. ResNet模型

假如神经网络的输入是x，期望输出是H(x)，如果我们把输入直接传到输出作为初始结果，那么我们需要学习的目标就是F(x)=H(x)-x。ResNet相当于将学习目标改变了，不再是学习一个完整的H(x)，而是学习输出和输入的差别，即残差F(x)。

ResNet模型中残差连接存在两种连接方式，即下图中的"实线"和"虚线"两种。

![connection](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/connection.png)

实线连接的两边都是`3*3*64`的特征图，他们通道数一致，所以采用计算方式：y=F(x)+x。

虚线连接的两边分别是`3*3*64`和`3*3*128`两种特征图，由于他们通道数不同，因此需要进行变换通道操作，即增加一个卷积操作，它采用的计算方式：y=F(x)+Wx，W是卷积操作。



ResNet论文中提出两种残差学习模块，分为两层和三层这两种学习模块，如下图所示。

![shortcut](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/shortcut.png)

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



### 二、RNN系列

#### 1. RNN

标准的RNN结构中，隐层的神经元之间也是带有权值的。也就是说，随着序列的不断推进，前面的隐层将会影响后面的隐层。如下图所示，O代表输出，X表示输入，S表示隐藏层向量。U是输入层到隐藏层的**权重矩阵**，V是隐藏层到输出层的**权重矩阵**。**权重矩阵**W就是**隐藏层**上一次的值作为这一次的输入的权重。**循环神经网络**的**隐藏层**的值s不仅仅取决于当前这次的输入x，还取决于上一次**隐藏层**的值s。

![rnn](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/rnn.jpg)

标准RNN的具有以下特点：
1、权值共享，图中的W全是相同的，U和V也一样。
2、每一个输入值都只与它本身的那条路线建立权连接，不会和别的神经元连接。

公式如下所示：

![rnn-equation](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/rnn-equation.jpg)

```python
Pytorch中有RNN系列的方法

input_size = 5
hidden_size = 10
batch_size = 3
n_layers = 2
seq_len = 8

rnn = nn.RNN(input_size, hidden_size, n_layers)

# 输入和初始隐藏变量
# seq_len * hidden * embedding(序列输入)
rnn_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏层)
rnn_h0 = torch.randn(n_layers, batch_size, hidden_size)

output, _ = rnn(x, h0)
print(output.shape)
# output shape (seq_len, batch, num_directions * hidden_size)
>> torch.Size([8, 3, 10])

Pytorch中还有pack_padded_sequence方法，用来处理不等长序列。pack_padded_sequence最主要的输入是输入数据以及每个样本的 seq_len 组成的 list。需要注意的是，我们必须把输入数据按照 seq_lenn 从大到小排列后才能送入 pack_padded_sequence。
```



#### 2. LSTM

（1）LSTM

LSTM（长短期记忆网络）是RNN的一种变体，RNN由于梯度消失的原因只能有短期记忆，LSTM网络通过精妙的门控制将短期记忆与长期记忆结合起来，并且一定程度上解决了梯度消失的问题。

所有 RNN 都具有一种重复神经网络模块的链式的形式。在标准的 RNN 中，这个重复的模块只有一个非常简单的结构，例如一个 tanh 层。LSTM模型同样如此，但是重复模块拥有一个不同的结构。整体上除了输出h随时间流动外，细胞状态c也在随时间流动，细胞状态c代表长期记忆，而输出h可以认为是短期记忆。

![LSTM](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/LSTM.png)

LSTM模型的关键就是细胞状态，它类似于传送带，长期记忆在上面传送，只进行一些线性操作，信息在上面流传保持不变相对容易。LSTM 通过精心设计的称作为“门”的结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个 pointwise 乘法操作。Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”。

在LSTM中第一步是决定我们从细胞状态中丢弃什么信息，这是通过一个遗忘门来实现的。该门会读取$h_{t-1}$和$x_{t}$，输出一个0-1之间的数来决定细胞状态$c_{t-1}$的保留程度。

下一步是确定多少新的信息放入到细胞状态中，即选择门。它由两块构成，其中sigmoid层，即输入门层决定新信息的更新程度。然后一个tanh层来产生一个新的候选向量，即当前需要新增的信息。最终我们将原先保留的信息和新增的信息共同合并成新的细胞状态$c_{t}$。**sigmoid函数选择更新内容，tanh函数创建更新候选。**

最终，我们需要确定输出什么值，即输出门。该输出会基于我们的细胞状态，先运行一个sigmoid层来确定细胞状态中哪部分将会被传输出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在 -1 到 1 之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定要输出的那部分。

这三个门虽然功能上不同，但在执行任务的操作上是相同的。他们都是**使用sigmoid函数作为选择工具，tanh函数作为变换工具**，这两个函数结合起来实现三个门的功能。

![LSTM-equation](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/LSTM-equation.jpeg)

（2）堆叠LSTM模型

通过对齐多个LSTM细胞（即多层LSTM网络结构），处理序列数据。如下图所示，是一个两层的LSTM模型，每层有4个细胞单元。通过这种方式，网络变得更加丰富，并捕获到更多的依赖项。

![img](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/multi-LSTM.png)

（3）双向LSTM模型

有时候分析序列数据，不仅仅需要依赖前向顺序，还需要考虑反向顺序。下图描述了该双向架构。

![img](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/biLSTM.jpg)

```python
input_size = 5
hidden_size = 10
batch_size = 3
n_layers = 2
seq_len = 8

lstm = nn.LSTM(input_size, hidden_size, n_layers)
# 双向lstm
bilstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=True)

# seq_len * batch * embedding(序列输入)
lstm_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏状态)
lstm_h0 = torch.randn(n_layers, batch_size, hidden_size)
# layer * batch * hidden(细胞状态)
lstm_c0 = torch.randn(n_layers, batch_size, hidden_size)

lstm_output, (hn, cn) = lstm(lstm_x, (lstm_h0, lstm_c0))
print(lstm_output.shape)
# output shape (seq_len, batch, num_directions * hidden_size)
>> torch.Size([8, 3, 10])

bilstm_output, _ = bilstm(lstm_x)
print(bilstm_output.shape)
>> torch.Size([8, 3, 20])
```



#### 3. GRU

GRU是LSTM中的一个变种，结构比LSTM简单一点。LSTM有三种门结构（遗忘门、输入门和输出门），而GRU只有两个门结构（更新门和重置门），另外GRU没有细胞状态c。相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

下图展示了GRU的结构。

![GRU](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/GRU.jpeg)

图中**z**t 和 **r**t 分别表示更新门 (红色) 和重置门 (蓝色)。重置门控制前一状态信息$h_{t-1}$传入候选状态（图中带波浪线的**h**t）的比例，重置门**r**t越小，表示添加到候选状态的信息越少。更新门用于控制前一状态的信息 **h**t-1 有多少保留到新状态 **h**t 中，当 (1- **z**t) 越大，保留的信息越多。GRU很聪明的一点就在于，**我们使用了同一个门控 ![[公式]](https://www.zhihu.com/equation?tex=z) 就同时可以进行遗忘和选择记忆（LSTM则要使用多个门控）**。

下图是GRU的更新公式。

![GRU-equation](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/GRU-equation.jpeg)

GRU输入输出的结构与普通的RNN相似，其中的内部思想与LSTM相似。

与LSTM相比，GRU内部少了一个”门控“，参数比LSTM少，但是却也能够达到与LSTM相当的功能。考虑到硬件的**计算能力**和**时间成本**，因而很多时候我们也就会选择更加”实用“的GRU啦。

```
input_size = 5
hidden_size = 10
batch_size = 3
n_layers = 2
seq_len = 8

gru = nn.GRU(input_size, hidden_size, n_layers)

# seq_len * batch * embedding(序列输入)
gru_x = torch.randn(seq_len, batch_size, input_size)
# layer * batch * hidden(隐藏状态)
gru_h0 = torch.randn(n_layers, batch_size, hidden_size)

gru_output, _ = self.gru(gru_x, gru_h0)
print(gru_output.shape)
>> torch.Size([8, 3, 10])
```



### 三、Transformer系列

![transformer](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/transformer.jpeg)

#### 1. Transformer

Transformer本身还是一个Encoder-Decoder模型，可以认为是一个Seq2Seq with attention的结构。

Encoder端由N个相同大模块堆叠而成。每个大模块由两个子模块组成，分别为多头self-attention模块和前馈神经网络模块。

Decoder端同样由N个相同大模块组成，其中每个大模块由三个子模块组成，分别为多头self-attention模块，多头Encoder-Decoder Attention模块和前馈神经网络结构。Decoder端训练和测试时接收的输入是不同的，训练时输入为上次输入加上输入序列后移一位的Ground Truth（比如一个新单词的词向量或者一段新音频特征），其中时间步为1时是一个特殊的token，用于对应任务设置的特定输入。**实际训练过程中一次性将目标序列的embedding都输入到第一个模块中，然后在多头attention模块中对输入序列进行mask**。测试时是挨个生成对应位置的输出。

Decoder端相比Encoder端增加一个子模块，即多头Encoder-Decoder attention交互模块，形式结构与self-attention模块一致，唯一不同是其Q、K、V矩阵的来源，其Q矩阵来源于下面子模块的输出，而K、V矩阵来自于整个Encoder端的输出，主要用于输出和输入产生关联，即attention。该交互模块跟Seq2Seq with attention中的机制一样。

各子模块：

（1）多头self-attention模块

Transformer中使用的attention公式是：

$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

可描述为**将query和key-value键值对的一组集合映射到输出**。多头self-attention模块，则是将Q、K、V通过参数矩阵映射后（给Q、K、V分别接一个全连接层），然后再做self-attention，最后再将所有结果拼接起来。

对应公式为：

$$MultiHead(Q,K,V)=Concat(head_1,head_2,...,head_h)W^O$$

$$where ~head_i=Attention(QW_i^Q,K_i^K,V_i^V)$$

其中$W_i^Q,W_i^K,W_i^V \in \mathbb{R}^{d_{model}\times d_k}$，$W^O \in \mathbb{R}^{hd_v \times d_{model}}$。

self-attention特点在于**无视token与token之间的距离直接计算依赖关系，从而能够学习到序列的内部结构**，另外由于不像RNN那样需要递归式执行，方便并行计算。self-attention更容易捕捉句子中长距离的依赖关系。**使用多头类似于CNN中使用多个卷积核的作用，形成多个子空间，有助于网络捕捉到更丰富的特征/信息**。

（2） Add & Norm模块

Add表示残差连接，Norm表示层归一化（LayerNorm）。因此每个子模块输出为LayerNorm(x+Sublayer(x))。

随着$d_k$的增大，q*k点积后结果也随之增大，这样会将softmax函数推入梯度非常小的区域，使之收敛困难，甚至可能出现梯度消失的情况。为了说明点积变大的原因，假设和的分量是具有均值 0 和方差 1 的独立随机变量，那么它们的点积均值为 0，方差为$d_k$，因此为了抵消这种影响，我们将点积缩放$\frac{1}{\sqrt{d_k}}$。

（3）Positional Encoding（位置编码）模块

self-attention模块将每个token之间产生联系，相对距离都变为1，而在序列中每个token之间的距离也具有重要意义，因此我们需要一个距离来表示每个token之间的位置关系。Transformer论文中使用了正弦余弦公式来生成位置编码。公式如下：

$$PE_(pos,2i)=sin(pos/10000^{2i/d_{model}})$$

$$PE_(pos,2i+1)=cos(pos/10000^{2i/d_{model}})$$

其中pos为位置，i为维度。通过三角变换可以将token之间的向量进行转化，这就引入了相对关系。**另外三角函数不受序列长度限制，也就是可以对比训练序列更长的序列进行表示。**

Transformer存在的两个问题：

（1）由于自注意力机制每次都需要计算所有词之间的注意力，其计算复杂度为输入长度的平方；

（2）Transformer需要事先设定输入长度，这对于长程关系捕捉具有一定限制，并且由于需要对输入文档进行分割会导致语意上的碎片化。

```python
1. position encoding
class PositionalEncoding(nn.Module):
    # 使用正弦余弦公式来计算位置编码
    # PE(pos, 2i)   = sin(pos/10000^(2i/d_model))
    # PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = dropout

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        # 函数 : e ^ (2i * -log(10000) / d), i是维度
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe = pe.unsqueeze(0)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))#torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
      
2. Attention
def attention(qurey, key, value, mask=None, dropout=None):
    # "点乘注意力机制实现"
    # 计算公式： att(Q,K,V) = softmax(QK^T/sqrt(d_k))V
    d_k = qurey.size(-1)
    scores = torch.matmul(qurey, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
      
3. MultiHead-Attention
```



Transformer相关问题

```
1. 不考虑多头，self-attention中词向量不乘QKV参数矩阵，会有什么问题。
self-attention核心是使用文本中的其他词来增强目标词的语义表示，从而更好的利用上下文信息，因此sequence中的每个词都会和所有词做点积计算相似度。如果不乘QKV参数矩阵，那么词的q,k,v完全是相同的，因此相等时点积最大，这就意味在Softmax后的加权平均中，该词所占的比重最大，无法有效利用上下文信息。而乘以QKV矩阵使得q，k，v都不同，一定程度上能有效减轻上述问题的影响。另外QKV矩阵也使得多头类似于CNN中的多个通道，能够捕捉更丰富的特征信息。

2. self-attention的时间复杂度。
self-attention分为三个步骤，相似度计算、softmax和加权平均。相似度计算的时间内复杂度是O(n^2*d)，n是序列长度，d是embedding长度。相似度计算相当于是大小为(n,d)和(d,n)的两个矩阵相乘，时间复杂度为O(n^2*d)，softmax时间复杂度为O(n^2)，加权评分相当于是(n,n)和(n,d)两个矩阵相乘，时间复杂度为O(n^2*d)，因此最终时间复杂度为O(n^2*d)。

3. Transformer在哪做了权重共享，为什么可以共享
（1）Encoder和Decoder间的Embedding层权重共享。
（2）Decoder中的Embeddig和FC层权重共享。
```



#### 2. Transformer-XL

相比于原生Transformer，Transformer-XL有以下两个变化：1）引入循环机制，使得新模型能够学习到更长的语义联系； 2）抛弃绝对位置，使用相对位置表示。与原生Transformer相比，Transformer-XL的另一个优势是它可以用于单词级或字符级的语言建模。

由于Transformer存在以下两个缺点，因此提出Transformer-XL：

- Transformer无法建模超过固定长度的依赖关系，对长文本编码效果差。
- Transformer把要处理的文本分割成等长的片段，通常不考虑句子（语义）边界，导致上下文碎片化(context fragmentation)。通俗来讲，一个完整的句子在分割后，一半在前面的片段，一半在后面的片段。

（1）引入循环机制

Transformer-XL使用分段的方式建模，但是不同于原生Transformer，它引入段与段之间的循环机制，使得当前段在建模时利用之前段段信息来实现长期依赖性。从思路来看，在原生Transformer基础上结合RNN的思想，只不过RNN是每个隐藏单元，而Transformer-XL是继承上一段段隐藏层，实现信息的传递。

![transformer-xl](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/transformer-xl.jpeg)

在训练阶段，处理后面的段时，每个隐藏层都会接收两个输入：

1. 该段的前面隐藏层的输出，与Transformer相同（上图的灰色线）。
2. 前面段的隐藏层的输出（上图的绿色线），可以使模型创建长期依赖关系。

对于某个段某一层的具体计算公式如下：

![xl-equation](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/xl-equation.jpeg)

其中，τ表示第几段，n表示第几层，h表示隐层的输出。SG(⋅)表示停止计算梯度，[hu∘hv]表示在长度维度上的两个隐层的拼接，W.是模型参数。和Transformer唯一关键的不同就在于Key和Value矩阵的计算上，即$k_{\tau +1}^n$和 $v_{\tau +1}^n$，它们基于的是扩展后的上下文隐层状态$\tilde{h}_{\tau +1}^{n-1}$进行计算，$h_{\tau}^{n-1}$是之前段的缓存。
测试阶段，每次可以前进一整个段，并利用之前段段数据来预测当前段的输出。

（2）相对位置编码

如果使用Transformer中的绝对位置信息来编码，那不同段中同一位置的位置编码是相同的，直观上这肯定是不对的，因此需要对token的位置进行区分。Transformer-XL提出了一种新的位置编码方式，即根据词之间的相对距离而不是绝对位置来编码。在Transformer中，计算查询$q_{i}^{T}$和键$k__{j}$之间的attention分数方式如下：

![abs-loc](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/abs-loc.jpeg)

其中$E_{x_{i}}$是token i的embedding，$E_{x_{j}}$是token j的embedding，$U_{i}$和$U_{j}$是位置向量。

Transformer-XL中使用了相对位置编码计算，公式如下：

![rel-loc](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/rel-loc.png)

有以下几项说明：

a. 在(b)和(d)这两项中，将所有绝对位置向量$U_{j}$都转为相对位置向量$R_{i-j}$，与Transformer相同，它是一个固定的编码向量，不需要学习；

b. 在(c)这一项中，将查询的$U_{i}^{T}W_{q}^{T}$向量转为一个需要学习的参数向量u，因为在考虑相对位置的时候，不需要查询绝对位置i，因此对于任意i，都采用相同的向量。同样在(d)项中将$U_{i}^{T}W_{q}^{T}$向量转为另一个需要学习的参数向量v。

c. 将键的权重矩阵W_{k}转为W_{K,E}和W_{K,R}，分别表示基于内容的权重矩阵和基于位置的权重矩阵。

d. (a)项表示不同token内容的注意力，这是不考虑位置信息的。(b)和(c)两项当前token对其他位置的关注程度。(d)项表示不同位置之间的关注程度。



#### 3. BERT

预训练语言模型主要分为两种，feature-based策略以及fine-tuning策略。

1. **feature-based**策略的代表模型为ELMo，它把预训练得到的“向量表示”作为训练下游任务的额外特征。训练下游任务时使用新的任务相关模型，并基于得到的特征来进行进一步的训练。
2. **fine-tuning**策略的代表模型为GPT，它则是在训练语言模型后，只改变极少的任务相关的层与参数，直接对下游任务训练整个原来的语言模型。

BERT使用的是后者，因为这种策略需要改变的参数量较少，迁移也较为简单。

###### BERT模型的几大核心贡献：

1. BERT揭示了语言模型的**深层双向学习能力**在任务中的重要性，特别是相比于同样在fine-tuning范畴内使用单向生成式训练的GPT以及浅层的双向独立训练并 concat 的ELMo，BERT的训练方法都有了很大的进步，BERT是通过改进训练目标来实现深层双向的语言模型训练，待会会单独介绍
2. BERT再次论证了**fine-tuning**的策略是可以有很强大的效果的，而且再也不需要为特定的任务进行繁重的结构设计。BERT也是使用fine-tuning策略的模型中第一个无论在句级别或在词级别都获得了state-of-art效果，胜过了不少专为相关任务设计的模型。
3. BERT在11个NLP任务上获得了state-of-art的效果，在SQuAD v1.1 问答任务上**超过人类水平**。

BERT在预训练时使用了两个非监督任务：

1. Masked LM（MLM）

   无论是单向的生成式语言模型，还是独立的left-to-right和right-to-left的进行拼接都不如真正的深层双向联合训练。但以标准的语言模型目标，没办法实现双向的训练，因为模型在预测某个单词时，会间接地在多层的上下文中看见“自己”，导致**泄露**。

   BERT提供的解决方案就是Mask LM 任务，它会随机mask掉一定比例的token，让它在训练的时候不在输入中出现，并把它们作为目标来训练，这样就可以防止泄露，mask的方式是把token替换成一个固定的token [MASK]。

   然而在实际使用时，因为MLM任务时语言模型的训练任务，其中[MASK]这种token只在训练时出现，在下游模型的fine-tuning时不会出现的，这就会导致预训练和fine-tuning时数据分布不一致。为了弥补这个问题，这15%应该被mask掉的token有80%的可能被替换成[MASK]，有10%的可能被替换成另外一个随机的token，另有10%的可能会维持原样不变。这样做，可以让Transformer的encoder无法去知道哪个token是要被预测的，也不知道哪个词被替换成随机词了，使它不得不对每一个token都形成一个较好的向量表示，没法取巧。

2. Next Sentence Prediction（NSP）

   很多任务，包括问答、自然语言推断等是基于理解**两句句子之间关系**的，不能直接被语言模型所建模，所以BERT还有另外一个二分类任务NSP来捕捉句子间的关系。在构造这个任务的数据集时，会有50%的概率，提供正样本，即某句句子和其下一句句子的组合，50%的概率在语料中选择任意一句句子构成负样本。这个任务相较MLM来说还是相当简单的。



下图展示了BERT、GPT以及ELMO模型对比：

![BERT](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT.png)

* ELMO模型核心组件是LSTM。最下方Embedding是词向量，中间是两层LSTM，分别代表从左向右和从右向左的双向网络；双向LSTM的输出在最上方拼接，形成包含上下文语义的向量表示。
* GPT模型的核心组件是Transformer。最下方的Embedding是token embedding和position embedding相加，token embedding的vocab是BPE算法得到的；中间是12层Transformer，语言模型为标准的单向条件概率，没有双向能力。
* BERT模型核心组件也是Transformer。最下方Embedding是token embedding、position embedding和segment embedding。token embedding的vocab为30000个左右词的Wordpiece embedding；中间的Transformer层有两种标准（12层和24层），因为MLM任务，所以BERT具有捕捉双向语义的能力。

BERT的输入也与GPT类似都用了[CLS]和[SEP]，相比之下在预训练和finetune都做了规范化和处理，以应对不同的任务。句子开头的token为[CLS]，结尾的token为[SEP]。如果输入仅有一句话，那规范化后的tokens是[CLS] [Sentence1] [SEP]，如果为两句话，那么规范后的tokens是 [CLS] [Sentence1] [SEP] [Sentence2] [SEP] 。另外，BERT模型还需要输入segment_id，以标识token的每一个位置是属于第一句话还是第二句话的，第一句话的位置上segment_id都是0，第二句话的位置都是1。

##### BERT下游应用

（1）Classification（分类问题）

![BERT-classification](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT-classification.jpeg)

使用[CLS]输出向量，加上softmax层做文本分类的下游任务。

（2）Questions & Answering

![BERT-QA](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT-QA.png)

根据段落和问题输入，查找段落中答案的起始和结束位置。

（3）Named Entity Recognition（NER）

![BERT-NER](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT-NER.png)

输出每个token对应的标签。

（4）Chat Bot（Intent Classification & Slot Filling）

![BERT-Chatbot](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT-Chatbot.jpeg)

使用[CLS]标签的输出向量做分类，用于意图识别；使用后面token的输出做NER，用于槽填充。

（5）Reading Comprehension

![BERT-RC](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/BERT-RC.png)

和QA类似。



BERT相关问题

```
1. 为什么BERT在第一句前会加一个[CLS]标志?
BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。
为什么选它呢，因为与文本中已有的其它词相比，这个无明显语义信息的符号会更公平地融合文本中各个词的语义信息，从而更好的表示整句话的语义。

2. BERT非线性的来源在哪里？
前馈层的gelu激活函数和self-attention。
```



#### 4. XLNet

采用了Permutation语言模型、以及使用了双流自注意力机制，并结合了Transformer-XL的相对位置编码。

- AR：Autoregressive Language Modeling（自回归模型）
- AE：Autoencoding Language Modeling（自编码模型）

![AR_AE](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/AR_AE.jpeg)

AR语言模型：指的是，依据前面（或后面）出现的tokens来预测当前时刻的token，代表有 ELMO， GPT等。

AE语言模型：通过上下文信息来预测被mask的token，代表有 BERT , Word2Vec(CBOW)  。

**AR 语言模型：**

- **缺点：** 它只能利用单向语义而不能同时利用上下文信息。ELMO 通过双向都做AR 模型，然后进行拼接，但从结果来看，效果并不是太好。
- **优点：** 对生成模型友好，天然符合生成式任务的生成过程。这也是为什么GPT能够编故事的原因。

**AE 语言模型：**

- **缺点：** 由于训练中采用了 `[MASK]` 标记，导致预训练与微调阶段不一致的问题。BERT独立性假设问题，即没有对被遮掩（Mask）的 token 之间的关系进行学习。此外对于生成式问题， AE 模型也显得捉襟见肘。
- **优点：** 能够很好的编码上下文语义信息（即考虑句子的双向信息）， 在自然语言理解相关的下游任务上表现突出。

ALNet是将两种语言模型结合起来使用。具体实现方式是，通过随机取一句话的一种排列，然后将末尾一定量的词给遮掩（和 BERT 里的直接替换 `[MASK]` 有些不同）掉，最后用 AR 的方式来按照这种排列依次预测被遮掩掉的词。

![PLM](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/PLM.jpeg)

通过随机取排列中的一种，就能很巧妙的通过AR的单向方式来学习双向信息了。可以通过设置Mask矩阵来设置排列方式。

![two-stream attention](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/two-stream attention.jpeg)



#### 5. ELECTRA

ELECTRA模型抛弃传统的MLM（Masked Language Model）任务，提出了全新的replaced token detection任务，使得模型保持性能同时降低模型参数量，提高模型运算速度。

Replaced token detection任务包含两个步骤：

1. mask一些input tokens，然后使用一个生成式网络预测被mask的token
2. 训练一个判别式网络来判断每一个token是否"虚假"。

下图显示RTD任务预训练的整体模型：

![RTD](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/RTD.png)

RTD的优化目标函数：

![RTD_loss](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/RTD_loss.png)

左边表示MLM的loss，右边表示判别模型的loss。预训练时，生成模型和判别模型同时训练。

Generator网络其实就是一个小型MLM，discriminator就是论文所说的ELECTRA模型。在预训练完成之后，generator被丢弃，而判别式网络会被保留用来做下游任务的基础模型。

尽管与GAN的训练目标很像，RTD任务与GAN存在一些关键性差异：

1. 如果generator正确还原了一些token，这些正确还原的token在discriminator部分会算作真实token。而在GAN中，只要是generator生成的token，就会被当作“虚假”token；
2. Generator的训练目标与MLM一样，而不是像GAN一样尽力去“迷惑” discriminator。对抗地训练generator是困难的，因为对文本来说，字词是离散的，无法用反向传播把discriminator的梯度传给generator。针对这一问题，作者尝试过使用强化学习来训练generator，但是效果并没有MLM的效果好；
3. GAN的输入是随机噪声，而ELECTRA的输入是真实文本。



#### 6. Star-Transformer

采用中心节点和卫星节点，中心节点初始值是卫星节点词向量的平均值（此处我认为有句向量借鉴词向量的思想在其中）。算法中参数更新分为两步：第一步为卫星节点的更新，第二步为中心节点的更新。两步的更新都是基于多头注意力机制。

![star-transformer](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/star-transformer.png)

对于卫星节点，计算多头注意力机制时只需考虑该节点状态与直接相邻节点，中心节点，该节点词向量和本节点上一时刻状态的信息交互。

因为中心节点担负着所有卫星节点之间的信息交互，因此中心节点在更新时须与自己上一时刻的信息和所有卫星节点进行信息交互。同时为了表示位置信息，在卫星节点中还必须拼接上表示位置信息的可学习的向量。

![star_learning](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/star_learning.jpeg)

该模型在使用中，针对序列的下游任务使用卫星节点的输出，而针对语言推理文本分类这种需要整个句子的任务则可以使用中心节点的输出。



