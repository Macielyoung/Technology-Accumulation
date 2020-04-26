## 常见知识点

1. 激活函数比较（sigmoid、tanh、RELU）

   一般tanh比sigmoid效果好一点(简单说明下，两者很类似，**tanh是rescaled的sigmoid，sigmoid输出都为正数，根据BP规则，某层的神经元的权重的梯度的符号和后层误差的一样**，也就是说，**如果后一层的误差为正，则这一层的权重全部都要降低，如果为负，则这一层梯度全部为负，权重全部增加，权重要么都增加，要么都减少，这明显是有问题的；tanh是以0为对称中心的，这会消除在权重更新时的系统偏差导致的偏向性**。当然这是启发式的，并不是说tanh一定比sigmoid的好)，ReLU也是很好的选择，最大的好处是，当tanh和sigmoid饱和时都会有梯度消失的问题，ReLU就不会有这个问题，而且计算简单，当然它会产生dead neurons。

2. Dropout的理解。

   - **一方面缓解过拟合，另一方面引入的随机性，可以平缓训练过程，加速训练过程，处理outliers**
   - **Dropout可以看做ensemble，特征采样，相当于bagging很多子网络**；训练过程中动态扩展拥有类似variation的输入数据集。（在单层网络中，类似折中Naive bayes(所有特征权重独立)和logistic regression(所有特征之间有关系)；
   - 一般对于越复杂的大规模网络，Dropout效果越好，是一个强regularizer！

3. 

