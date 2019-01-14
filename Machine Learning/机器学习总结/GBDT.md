## GBDT(Gradient Boosting Decision Tree)

**基本介绍**

GBDT是一种迭代的决策树算法，它通过构造一组弱的学习器（树），并把多棵决策树的结果累加起来作为最终的预测输出。该算法将决策树和集成思想进行了有效的组合。GBDT的思想使其能够发现多种有区分性的特征及特征组合，是泛化能力较强的算法。



**提升树**

提升方法采用加法模型（即基函数的线性组合）与前向分布算法。以决策树为基函数的提升方法称为提升树。对分类问题构建的决策树是二叉分类树，对回归问题构建决策树是二叉回归树。提升树是迭代多棵回归树来共同决策。当采用平方误差损失函数时，每一棵回归树学习的是之前所有树的结论和残差，拟合得到一个当前的残差回归树，残差的意义如公式：残差 = 真实值 - 预测值 。具体例子可参考蓝皮书149-151页。



**GBDT简介**

在GBDT的迭代中，假设前一轮得到的抢学习器为`ft−1(x)`，对应的损失函数则为`L(y,ft−1(x))`。因此新一轮迭代的目的就是找到一个弱分类器`ht(x)`，使得损失函数`L(y,ft−1(x)+ht(x))`达到最小。**因此问题的关键就在于对损失函数的度量，这也正是难点所在。**

针对这一问题，Freidman提出了梯度提升算法：利用最速下降的近似方法，即利用损失函数的负梯度在当前模型的值

![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/residual.png)

作为回归问题中提升树算法的残差的近似值（与其说负梯度作为残差的近似值，不如说残差是负梯度的一种特例，拟合一个回归树），这就是梯度提升决策树。



**算法步骤**

![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt.jpg)

具体过程：

1）初始化弱分类器，估计使损失函数极小化的一个常数值，此时树仅有一个根结点

![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt1.jpg)

2）对迭代轮数1,2,···,M 

- a）对i=1,2,···,N，计算损失函数的负梯度值在当前模型的值，将它作为残差的估计。即

  ![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt2.jpg)

  对于平方损失函数，它就是通常所说的残差；对于一般损失函数，它就是残差的近似值。

- b）对r_{mi}拟合一个回归树，得到第m棵树的叶结点区域Rmj，j=1,2,···,J.

- c）对j=1,2,···,J计算

  ![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt3.jpg)

  即利用线性搜索估计叶结点区域的值，使损失函数极小化

- d）更新回归树

  ![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt4.jpg)

3）得到输出的最终模型

![](https://github.com/Macielyoung/Technology-Accumulation/blob/master/Machine%20Learning/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%80%BB%E7%BB%93/pic/gbdt5.jpg)

