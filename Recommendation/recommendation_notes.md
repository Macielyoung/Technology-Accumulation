## 深度学习推荐系统实战笔记

### 整体框架

![img](images/066c5f56f4e0a5e8d4648e0cfb85e72e.png)



### 推荐系统经典架构长啥样

1. 推荐系统要解决的问题

   在“信息过载”的情况下，用户如何高效获取感兴趣的信息。

   推荐系统要被处理的问题可以被形式化定义为：**对于某个用户U（User），在特定场景C（Context）下，针对海量的“物品”信息构建一个函数 ，预测用户对特定候选物品I（Item）的喜好程度，再根据喜好程度对所有候选物品进行排序，生成推荐列表的问题。**

   

2. 推荐系统逻辑架构

   ![img](images/c75969c5fcc6e5e374a87d4b4b1d5d07.png)

3. 推荐系统中着重需要解决的两类问题

   （1） 一类问题与数据和信息相关，即“用户信息”“物品信息”“场景信息”分别是什么？如何存储、更新和处理数据？

   （2） 另一类问题与推荐系统算法和模型相关，即推荐系统模型如何训练、预测，以及如何达成更好的推荐效果？

   

4. 工业级推荐系统的技术架构

   （1） “数据和信息”部分逐渐发展为推荐系统中融合了数据离线批处理、实时流处理的数据流框架；

   （2） “算法和模型”部分进一步细化为推荐系统中，集训练、评估、部署、线上推断为一体的模型框架。

   ![img](images/a87530cf45fb76480bf5b60b9feb60c1.png)

   

5. 大数据平台加工后的数据出口

   （1）生成推荐系统模型所需的样本数据，用于算法模型的训练和评估；

   （2）生成推荐系统模型服务（Model Serving）所需的“用户特征”，“物品特征”和一部分“场景特征”，用于推荐系统的线上推断。

   （3）生成系统监控，商业智能系统所需的统计型数据。

   

6. 深度学习对推荐系统模型的典型应用

   （1）深度学习中 Embedding 技术在召回层的应用。作为深度学习中非常核心的 Embedding 技术，将它应用在推荐系统的召回层中，做相关物品的快速召回，已经是业界非常主流的解决方案了。

   （2）不同结构的深度学习模型在排序层的应用。排序层（也称精排层）是影响推荐效果的重中之重，也是深度学习模型大展拳脚的领域。深度学习模型的灵活性高，表达能力强的特点，这让它非常适合于大数据量下的精确排序。深度学习排序模型毫无疑问是业界和学界都在不断加大投入，快速迭代的部分。

   （3）增强学习在模型更新、工程模型一体化方向上的应用。增强学习可以说是与深度学习密切相关的另一机器学习领域，它在推荐系统中的应用，让推荐系统可以在实时性层面更上一层楼。



### 推荐系统特征工程

1. 什么是特征工程

   推荐系统就是利用“用户信息”，“物品信息”，“场景信息”这三大部分有价值数据，通过构建推荐模型得出推荐列表的工程系统。特征工程就是利用工程手段从以上信息中提取的过程。特征工程的原则：**尽可能地让特征工程抽取出的一组特征，能够保留推荐环境及用户行为过程中的所有“有用“信息，并且尽量摒弃冗余信息。**

   

2. 推荐系统中的常用特征

   （1）用户行为数据。用户的潜在兴趣、用户对物品的真实评价都包含在用户的行为历史中。用户行为在推荐系统中一般分为显性反馈行为（Explicit Feedback）和隐性反馈行为（Implicit Feedback）两种。在当前的推荐系统特征工程中，隐性反馈行为越来越重要，主要原因是显性反馈行为的收集难度过大，数据量小。所以，能够反映用户行为特点的隐性反馈是目前特征挖掘的重点。

   ![img](images/7523075958d83e9bd08966b77ea23706.png)

   （2）用户关系数据。互联网本质上就是人与人、人与信息之间的连接。如果说用户行为数据是人与物之间的“连接”日志，那么用户关系数据就是人与人之间连接的记录。用户关系数据也可以分为“显性”和“隐性”两种，或者称为“强关系”和“弱关系”。用户与用户之间可以通过“关注”“好友关系”等连接建立“强关系”，也可以通过“互相点赞”“同处一个社区”，甚至“同看一部电影”建立“弱关系”。

   （3）属性、标签类数据。它们本质上都是直接描述用户或者物品的特征。属性和标签的主体可以是用户，也可以是物品。它们的来源非常多样，大体上包含以下几类。

   ![img](images/ba044e0033b513d996633de77e11f969.png)

   （4）内容类数据。内容类数据可以看作属性标签型特征的延伸，同样是描述物品或用户的数据，但相比标签类特征，内容类数据往往是大段的描述型文字、图片，甚至视频。

   （5）场景信息（上下文信息）。它是描述推荐行为产生的场景的信息。最常用的上下文信息是“时间”和通过 GPS、IP 地址获得的“地点”信息。在实际的推荐系统应用中，我们更多还是利用时间、地点、推荐页面这些易获取的场景特征。



### Spark解决特征处理

1. 类别型特征处理

   我们进行特征处理的目的，是把所有的特征全部转换成一个数值型的特征向量，对于数值型特征，这个过程非常简单，直接把这个数值放到特征向量上相应的维度上就可以了。但是对于类别、ID 类特征，这里我们就要用到 One-hot 编码（也被称为独热编码），它是将类别、ID 型特征转换成数值向量的一种最典型的编码方式。它通过把所有其他维度置为 0，单独将当前类别或者 ID 对应的维度置为 1 的方式生成特征向量。

   Spark使用机器学习库MLlib来完成One-hot特征的处理。最主要的步骤是，我们先创建一个负责 One-hot 编码的转换器，OneHotEncoderEstimator，然后通过它的 fit 函数完成指定特征的预处理，并利用 transform 函数将原始特征转换成 One-hot 特征。实现代码如下：

   ```scala
   
   def oneHotEncoderExample(samples:DataFrame): Unit ={
     //samples样本集中的每一条数据代表一部电影的信息，其中movieId为电影id
     val samplesWithIdNumber = samples.withColumn("movieIdNumber", col("movieId").cast(sql.types.IntegerType))
   
   
     //利用Spark的机器学习库Spark MLlib创建One-hot编码器
     val oneHotEncoder = new OneHotEncoderEstimator()
       .setInputCols(Array("movieIdNumber"))
       .setOutputCols(Array("movieIdVector"))
       .setDropLast(false)
   
   
     //训练One-hot编码器，并完成从id特征到One-hot向量的转换
     val oneHotEncoderSamples = oneHotEncoder.fit(samplesWithIdNumber).transform(samplesWithIdNumber)
     //打印最终样本的数据结构
     oneHotEncoderSamples.printSchema()
     //打印10条样本查看结果
     oneHotEncoderSamples.show(10)
   
   _（参考 com.wzhe.sparrowrecsys.offline.spark.featureeng.FeatureEngineering__中的oneHotEncoderExample函数）_
   ```

   针对多标签特征来说，转变为对应的Multi-Hot编码（多热编码）即可。

   

2. 数值型特征处理

   数值型的特征存在两方面问题，一是特征的尺度，一是特征的分布。

   特征尺度易于理解。比如电影推荐中评价次数fr和平均评分fs这两个特征，评价次数理论上是一个数值无上限的特征，而对评分来说，由于采取五分制，所以取值范围在[0,5]之间。由于 fr 和 fs 两个特征的尺度差距太大，如果我们把特征的原始数值直接输入推荐模型，就会导致这两个特征对于模型的影响程度有显著的区别。如果模型中未做特殊处理的话，fr 这个特征由于波动范围高出 fs 几个量级，可能会完全掩盖 fs 作用，这当然是我们不愿意看到的。为此我们希望把两个特征的尺度拉平到一个区域内，通常是[0,1]范围，这就是所谓归一化。

   归一化虽然能够解决特征取值范围不统一的问题，但无法改变特征值的分布。比如电影评分中，大量集中在3.5分附近，越靠近3.5分密度越大。这对于模型学习来说也不是一个好的现象，因为特征的区分度并不高。我们经常会用**分桶（Bucketing）**的方式来解决特征值分布极不均匀的问题。所谓“分桶”，就是将样本按照某特征的值从高到低排序，然后按照桶的数量找到分位数，将样本分到各自的桶中，再用桶 ID 作为特征值。

   在 Spark MLlib 中，分别提供了两个转换器 MinMaxScaler 和 QuantileDiscretizer，来进行归一化和分桶的特征处理。它们的使用方法和之前介绍的 OneHotEncoderEstimator 一样，都是先用 fit 函数进行数据预处理，再用 transform 函数完成特征转换。下面代码利用这两个转换器完成特征归一化和分桶的过程。

   ```scala
   
   def ratingFeatures(samples:DataFrame): Unit ={
     samples.printSchema()
     samples.show(10)
   
   
     //利用打分表ratings计算电影的平均分、被打分次数等数值型特征
     val movieFeatures = samples.groupBy(col("movieId"))
       .agg(count(lit(1)).as("ratingCount"),
         avg(col("rating")).as("avgRating"),
         variance(col("rating")).as("ratingVar"))
         .withColumn("avgRatingVec", double2vec(col("avgRating")))
   
   
     movieFeatures.show(10)
   
   
     //分桶处理，创建QuantileDiscretizer进行分桶，将打分次数这一特征分到100个桶中
     val ratingCountDiscretizer = new QuantileDiscretizer()
       .setInputCol("ratingCount")
       .setOutputCol("ratingCountBucket")
       .setNumBuckets(100)
   
   
     //归一化处理，创建MinMaxScaler进行归一化，将平均得分进行归一化
     val ratingScaler = new MinMaxScaler()
       .setInputCol("avgRatingVec")
       .setOutputCol("scaleAvgRating")
   
   
     //创建一个pipeline，依次执行两个特征处理过程
     val pipelineStage: Array[PipelineStage] = Array(ratingCountDiscretizer, ratingScaler)
     val featurePipeline = new Pipeline().setStages(pipelineStage)
   
   
     val movieProcessedFeatures = featurePipeline.fit(movieFeatures).transform(movieFeatures)
     //打印最终结果
     movieProcessedFeatures.show(10)
   
   _（参考 com.wzhe.sparrowrecsys.offline.spark.featureeng.FeatureEngineering中的ratingFeatures函数）_
   ```

   

3. 特征处理总结

   特征处理没有固定模式，上面列的只是一些常见处理方法，在实际应用中，我们需要多种尝试，找到最能提升模型效果的处理方式。

![img](images/b3b8c959df72ce676ae04bd8dd987e7b.png)



4. Spark中几个常用正则归一化函数

   ```
   Normalizer: 计算p范数，然后该样本中每个元素除以该范数。l1: 每个样本中每个元素绝对值的和，l2: 每个样本中每个元素的平方和开根号，lp: 每个样本中每个元素的p次方和的p次根，默认用l2范数。
   
   StandardScaler: 数据标准化，(xi - u) / σ 【u:均值，σ：方差】当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布）。
   
   RobustScaler: (xi - median) / IQR 【median是样本的中位数，IQR是样本的 四分位距：根据第1个四分位数和第3个四分位数之间的范围来缩放数据】。
   
   MinMaxScaler: 数据归一化，(xi - min(x)) / (max(x) - min(x)) ;当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到 [0,1]之间。
   ```