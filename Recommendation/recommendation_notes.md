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

   ```tex
   Normalizer: 计算p范数，然后该样本中每个元素除以该范数。l1: 每个样本中每个元素绝对值的和，l2: 每个样本中每个元素的平方和开根号，lp: 每个样本中每个元素的p次方和的p次根，默认用l2范数。
   
   StandardScaler: 数据标准化，(xi - u) / σ 【u:均值，σ：方差】当数据(x)按均值(μ)中心化后，再按标准差(σ)缩放，数据就会服从为均值为0，方差为1的正态分布（即标准正态分布）。
   
   RobustScaler: (xi - median) / IQR 【median是样本的中位数，IQR是样本的 四分位距：根据第1个四分位数和第3个四分位数之间的范围来缩放数据】。
   
   MinMaxScaler: 数据归一化，(xi - min(x)) / (max(x) - min(x)) ;当数据(x)按照最小值中心化后，再按极差（最大值 - 最小值）缩放，数据移动了最小值个单位，并且会被收敛到 [0,1]之间。
   ```



### 推荐系统中的Embedding技术

1. embedding在特征工程中的作用

   （1）Embedding是处理稀疏特征的利器。因为推荐场景中的类别、ID 型特征非常多，大量使用 One-hot 编码会导致样本特征向量极度稀疏，而深度学习的结构特点又不利于稀疏特征向量的处理，因此几乎所有深度学习推荐模型都会由 Embedding 层负责将稀疏高维特征向量转换成稠密低维特征向量。

   （2）Embedding 可以融合大量有价值信息，本身就是极其重要的特征向量。相比由原始信息直接处理得来的特征向量，Embedding 的表达能力更强，特别是 Graph Embedding 技术被提出后，Embedding 几乎可以引入任何信息进行编码，使其本身就包含大量有价值的信息，所以通过预训练得到的 Embedding 向量本身就是极其重要的特征向量。

   

2. Word2vec模型结构

   Word2vec模型本质上是一个三层的神经网络。如下图所示。

   ![img](images/9997c61588223af2e8c0b9b2b8e77139.png)

   它的输入层和输出层的维度都是 V，这个 V 其实就是语料库词典的大小。假设语料库一共使用了 10000 个词，那么 V 就等于 10000。这里的输入向量自然就是由输入词转换而来的 One-hot 编码向量，输出向量则是由多个输出词转换而来的 Multi-hot 编码向量，显然，基于 Skip-gram 框架的 Word2vec 模型解决的是一个多分类问题。输入向量矩阵 WVxN 的每一个行向量对应的就是我们要找的“词向量”。



### Graph Embedding和Embedding应用

1. DeepWalk

   我们基于原始的用户行为序列来构建物品关系图，然后采用随机游走的方式随机选择起始点，重新产生物品序列，最后将这些随机游走生成的物品序列输入 Word2vec 模型，生成最终的物品 Embedding 向量。

   

2. Node2Vec

   Node2vec 相比于 Deep Walk，增加了随机游走过程中跳转概率的倾向性。如果倾向于宽度优先搜索，则 Embedding 结果更加体现“结构性”。如果倾向于深度优先搜索，则更加体现“同质性”。

   

3. Embedding应用于推荐系统的特征工程中

   Embedding在推荐系统中应用方式主要有三种，分别是直接应用、预训练应用和End2End应用。

   “直接应用”是直接利用Embedding向量的相似性实现某些推荐系统的功能。典型功能有，利用物品Embedding间的相似性实现相似物品推荐，利用物品Embedding和用户Embedding的相似性实现“猜你喜欢”等经典推荐功能，还可以利用物品Embedding实现推荐系统中的召回层等。

   “预训练应用”指的是在我们预先训练好物品和用户的Embedding之后，不直接应用，而是把这些Embedding向量作为特征向量的一部分，跟其余的特征向量拼接起来，作为推荐模型的输入参与训练。这样做能够更好的把其他特征引入进来，让推荐模型做出更为全面切准确的预测。

   “End2End应用”也就是端对端训练，是指把Embedding的训练和深度学习推荐模型结合起来，采用统一的、端对端的方式一起训练，直接得到包含Embedding层的推荐模型。这种方式非常流行，比如下图就展示了三个包含 Embedding 层的经典模型，分别是微软的 Deep Crossing，UCL 提出的 FNN 和 Google 的 Wide&Deep。

   ![img](images/e9538b0b5fcea14a0f4bbe2001919978.png)

   

4. Embedding预训练和End2End两种方式的优缺点。

   Embedding预训练的优点：1.更快。因为对于End2End的方式，Embedding层的优化还受推荐算法的影响，这会增加计算量。2.难收敛。推荐算法是以Embedding为前提的，在端到端的方式中，embedding层包含大量参数，可能无法有效收敛。
   Embedding端到端的优点：可能收敛到更好的结果。端到端因为将Embedding和推荐算法连接起来训练，那么Embedding层可以学习到最有利于推荐目标的Embedding结果。

   

### Spark生成Item2vec和Graph Embedding

1. spark生成Item2vec

   ```scala
   def trainItem2vec(samples : RDD[Seq[String]]): Unit ={
       //设置模型参数(向量维度、滑动窗口大小和训练迭代次数)
       val word2vec = new Word2Vec()
       .setVectorSize(10)
       .setWindowSize(5)
       .setNumIterations(10)
   
   
     //训练模型
     val model = word2vec.fit(samples)
   
   
     //训练结束，用模型查找与item"592"最相似的20个item
     val synonyms = model.findSynonyms("592", 20)
     for((synonym, cosineSimilarity) <- synonyms) {
       println(s"$synonym $cosineSimilarity")
     }
    
     //保存模型
     val embFolderPath = this.getClass.getResource("/webroot/sampledata/")
     val file = new File(embFolderPath.getPath + "embedding.txt")
     val bw = new BufferedWriter(new FileWriter(file))
     var id = 0
     //用model.getVectors获取所有Embedding向量
     for (movieId <- model.getVectors.keys){
       id+=1
       bw.write( movieId + ":" + model.getVectors(movieId).mkString(" ") + "\n")
     }
     bw.close()
   ```



### 特征工程及深度学习一些问题

1. 对训练数据中的某项特征进行平方或者开方，是为了改变训练数据的分布。训练数据的分布被改变后，训练出来的模型岂不是不能正确拟合训练数据了？

   对训练数据中的某个特征进行开方或者平方操作，本质上是改变了特征的分布，并不是训练数据的分布。特征的分布和训练数据的分布没有本质的联系，只要你不改变训练数据 label 的分布，最终预测出的结果都应该是符合数据本身分布的。因为你要预测的是 label，并不是特征本身。而且在最终的预测过程中，这些开方、平方的特征处理操作是在模型推断过程中复现的，本质上可以看作是模型的一部分，所以不存在改变数据分布的问题。

   

2. 为什么深度学习的结构特点不利于稀疏特征向量的处理呢？

   一方面，如果我们深入到神经网络的梯度下降学习过程就会发现，特征过于稀疏会导致整个网络的收敛非常慢，因为每一个样本的学习只有极少数的权重会得到更新，这在样本数量有限的情况下会导致模型不收敛。另一个方面，One-hot 类稀疏特征的维度往往非常地大，可能会达到千万甚至亿的级别，如果直接连接进入深度学习网络，那整个模型的参数数量会非常庞大，这对于一般公司的算力开销都是吃不消的。

   所以基于上面两个原因，我们往往先通过 Embedding 把原始稀疏特征稠密化，然后再输入复杂的深度学习网络进行训练，这相当于把原始特征向量跟上层复杂深度学习网络做一个隔离。



### 线上提供高并发的推荐服务

1. 高并发推荐服务整体架构

   高并发推荐服务的整体架构主要由三个重要机制支撑，它们分别是负载均衡、缓存、推荐服务降级机制。

   通过增加服务器来分担独立节点的压力，同时合理分配任务，以达到按能力分配和高效率分配的目的，分配任务的机器称之为“负载均衡服务器”。

   缓存是指同一个用户多次请求同样的推荐服务时，可以把第一次请求时的推荐结果缓存起来，后续请求时直接返回缓存中的结果；另外对于新用户，可以按照一些规则预先缓存几类新用户的推荐列表。在一个成熟的工业级推荐系统中，合理的缓存策略甚至能够阻挡掉 90% 以上的推荐请求，大大减小推荐服务器的计算压力。

   服务降级就是抛弃原本的复杂逻辑，采用最保险、最简单、最不消耗资源的降级服务来渡过特殊时期。比如对于推荐服务来说，我们可以抛弃原本的复杂推荐模型，采用基于规则的推荐方法来生成推荐列表，甚至直接在缓存或者内存中提前准备好应对故障时的默认推荐列表，做到“0”计算产出服务结果，这些都是服务降级的可行策略。

   总之，“负载均衡”提升服务能力，“缓存”降低服务压力，“服务降级”机制保证故障时刻的服务不崩溃，压力不传导，这三点可以看成是一个成熟稳定的高并发推荐服务的基石。



### 推荐系统特征的存储

1. SparrowRecsys推荐系统数据存储方式

   该推荐系统中主要包含以下几类特征：

   ![img](images/d9cf4b8899ff4442bc7cd87f502a9c2a.png)

   （1）用户特征的总数比较大，它们很难全部载入到服务器内存中，所以我们把用户特征载入到 Redis 之类的内存数据库中是合理的。

   （2）物品特征的总数比较小，而且每次用户请求，一般只会用到一个用户的特征，但为了物品排序，推荐服务器需要访问几乎所有候选物品的特征。针对这个特点，我们完全可以把所有物品特征阶段性地载入到服务器内存中，大大减少 Redis 的线上压力。

   （3）我们还要找一个地方去存储特征历史数据、样本数据等体量比较大，但不要求实时获取的数据。这个时候分布式文件系统（单机环境下以本机文件系统为例）往往是最好的选择，由于类似 HDFS 之类的分布式文件系统具有近乎无限的存储空间，我们可以把每次处理的全量特征，每次训练的 Embedding 全部保存到分布式文件系统中，方便离线评估时使用。

   最终特征存储方式总结如下：

   ![img](images/34958066e8704ea2780d7f8007e18463.png)

   

2. Redis数据库基本知识

   Redis 是当今业界最主流的内存数据库，那在使用它之前，我们应该清楚 Redis 的两个主要特点。

   **一是所有的数据都以 Key-value 的形式存储。** 其中，Key 只能是字符串，value 可支持的数据结构包括 string(字符串)、list(链表)、set(集合)、zset(有序集合) 和 hash(哈希)。这个特点决定了 Redis 的使用方式，无论是存储还是获取，都应该以键值对的形式进行，并且根据你的数据特点，设计值的数据结构。

   **二是所有的数据都存储在内存中，磁盘只在持久化备份或恢复数据时起作用。**这个特点决定了 Redis 的特性，一是 QPS 峰值可以很高，二是数据易丢失，所以我们在维护 Redis 时要充分考虑数据的备份问题，或者说，不应该把关键的业务数据唯一地放到 Redis 中。但对于可恢复，不关乎关键业务逻辑的推荐特征数据，就非常适合利用 Redis 提供高效的存储和查询服务。





