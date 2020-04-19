## 1. CRF算法

CRF算法常用于命名实体识别任务（NER），下图展示了CRF一个使用案例。

![CRF](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/CRF.png)

图中说明BiLSTM层的输出是每个标签的分数。例如，对于w0, BiLSTM节点的输出为1.5 (B-Person)、0.9 (I-Person)、0.1 (B-Organization)、0.08 (I-Organization)和0.05 (O)，这些分数将作为CRF层的输入。然后，将BiLSTM层预测的所有分数输入CRF层。在CRF层中，选择预测得分最高的标签序列作为最佳答案。

CRF层可以向最终的预测标签添加一些约束，以确保它们是有效的。这些约束可以由CRF层在训练过程中从训练数据集自动学习。



CRF层的损失函数中，我们有两种类型的分数，这两个分数是CRF层的关键概念。

第一个是emission得分，他们来自于bilstm层，如上图所示，标记为B-Person的w0的分数为1.5。

第二个是transition得分。因此，我们有一个transition得分矩阵，它存储了所有标签之间的所有得分。为了使transition评分矩阵更健壮，我们将添加另外两个标签，START和END。START是指一个句子的开头，而不是第一个单词。END表示句子的结尾。下图是一个transition得分矩阵案例。

![transition](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/transition.jpeg)

Transition矩阵是BiLSTM-CRF模型的一个参数。在训练模型之前，可以随机初始化矩阵中的所有transition分数。所有的随机分数将在你的训练过程中自动更新。换句话说，CRF层可以自己学习这些约束。我们不需要手动构建矩阵。随着训练迭代次数的增加，分数会逐渐趋于合理。

CRF损失函数由真实路径得分和所有可能路径的总得分组成。在所有可能的路径中，真实路径的得分应该是最高的。例如我们有以下几个标签。

![CRF-labels](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/CRF-labels.jpeg)

我们还是有一个5个单词的句子。可能的路径是：

- 1) START B-Person B-Person B-Person B-Person B-Person END
- 2) START B-Person I-Person B-Person B-Person B-Person END
- …
- **10) START B-Person I-Person O B-Organization O END**
- …
- N) O O O O O O O

假设每条可能的路径都有一个分数Pi，并且总共有N条可能的路径，所有路径的总分数是Ptotal = P1+P2+…+PN。如果我们说第10条路径是真正的路径，换句话说，第10条路径是我们的训练数据集提供的黄金标准标签。在所有可能的路径中，得分P10应该是百分比最大的。

![CRF-loss](/Users/maciel/Documents/gitprojet/Technology-Accumulation/NLP/pic/CRF-loss.png)

在训练过程中，CRF损失函数只需要两个分数：真实路径的分数和所有可能路径的总分数。**所有可能路径的分数中，真实路径分数所占的比例会逐渐增加**。

计算**实际路径**分数$e^{Si}$非常简单。Si主要由两部分组成，EmissionScore + TransitionScore。

假如我们的真实路径为**“START B-Person I-Person O B-Organization O END”**，则

**EmissionScore**=x0,START+x1,B-Person+x2,I-Person+x3,O+x4,B-Organization+x5,O+x6,END

x{index, label}是第index个单词被label标注的单词，这些得分来自于之前的Bilstm的输出，而START和END标记得分可以设为0。

**TransitionScore**=tSTART->B-Person+tB-Person->I-Person+tI-Person->O+tO->B-Organization+tB-Organization->O+tO->END

t{label1->label2}是从label1转移到label2的分数，他们来自于CRF层，是需要迭代计算的参数。

下面需要计算所有可能路径的总得分。最简单的方法是列举所有可能组合方式，然而这非常低效。其实这个就是一个简单的排列组合问题，假如有n个标签，m个单词，那么所有组合方式有n^m种组合方式，不过每个标签在不同位置对应的得分并不相同。这个问题可以使用动态规划算法来实现。

计算损失函数方法：

```Python
def _forward_alg(self, feats):
    # Do the forward algorithm to compute the partition function
    init_alphas = torch.full((1, self.tagset_size), -10000.)
    # START_TAG has all of the score.
    init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

    # Wrap in a variable so that we will get automatic backprop
    forward_var = init_alphas

    # Iterate through the sentence
    for feat in feats:
        alphas_t = []  # The forward tensors at this timestep
        for next_tag in range(self.tagset_size):
            # broadcast the emission score: it is the same regardless of
            # the previous tag
            emit_score = feat[next_tag].view(
              1, -1).expand(1, self.tagset_size)
            # the ith entry of trans_score is the score of transitioning to
            # next_tag from i
            trans_score = self.transitions[next_tag].view(1, -1)
            # The ith entry of next_tag_var is the value for the
            # edge (i -> next_tag) before we do log-sum-exp
            next_tag_var = forward_var + trans_score + emit_score
            # The forward variable for this tag is log-sum-exp of all the
            # scores.
            alphas_t.append(log_sum_exp(next_tag_var).view(1))
        forward_var = torch.cat(alphas_t).view(1, -1)
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    alpha = log_sum_exp(terminal_var)
    return alpha
  
# compute log sum exp in numerically stable way for the forward algorithm
def log_sum_exp(vec):   #vec是1*5, type是Variable
    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，
    # max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 里面先做减法，减去最大值可以避免e的指数次，计算机上溢
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

# loss 函数
def neg_log_likelihood(self, sentence, tags):
    # feats: 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
    feats = self._get_lstm_features(sentence)
    forward_score = self._forward_alg(feats) 
    gold_score = self._score_sentence(feats, tags)
    return forward_score - gold_score
      
# 根据真实的标签算出的一个score，
# 这与上面的def _forward_alg(self, feats)共同之处在于：
# 两者都是用的随机转移矩阵算的score
# 不同地方在于，上面那个函数算了一个最大可能路径，但实际上可能不是真实的各个标签转移的值
# 例如：真实标签是N V V,但是因为transitions是随机的，所以上面的函数得到其实是N N N这样，
# 两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，使得转移矩阵逼近
#真实的“转移矩阵”
# 得到gold_seq tag的score 即根据真实的label 来计算一个score，
# 但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
def _score_sentence(self, feats, tags): #feats 11*5  tag 11 维
    # gives the score of a provied tag sequence
    score = torch.zeros(1)

    # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了
    tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

  	for i, feat in enumerate(feats):
        # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
        # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
        # 对应B, I, E, START_TAG, END_TAG, 取对应标签的值
        # transition【j,i】 就是从i ->j 的转移概率值
        score = score + \
            self.transitions[tags[i+1], tags[i]] + feat[tags[i + 1]]
    score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
    return score
  
# 维特比解码， 实际上就是在预测的时候使用了， 输出得分与路径值
# 预测序列的得分
def _viterbi_decode(self, feats):
    backpointers = []

    # initialize the viterbi variables in long space
    init_vvars = torch.full((1, self.tagset_size), -10000.)
    init_vvars[0][self.tag_to_ix[START_TAG]] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    forward_var = init_vvars
    for feat in feats:
        bptrs_t = [] # holds the backpointers for this step
        viterbivars_t = [] # holds the viterbi variables for this step

        for next_tag in range(self.tagset_size):
            # next-tag_var[i] holds the viterbi variable for tag i
            # at the previous step, plus the score of transitioning
            # from tag i to next_tag.
            # we don't include the emission scores here because the max
            # does not depend on them(we add them in below)
            # 其他标签（B,I,E,Start,End）到标签next_tag的概率
            next_tag_var = forward_var + self.transitions[next_tag]
            best_tag_id = argmax(next_tag_var)
            bptrs_t.append(best_tag_id)
            viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # now add in the emssion scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t) # bptrs_t有５个元素

    # transition to STOP_TAG
    # 其他标签到STOP_TAG的转移概率
    terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
    best_tag_id = argmax(terminal_var)
    path_score = terminal_var[0][best_tag_id]

    # follow the back pointers to decode the best path
    best_path = [best_tag_id]
    for bptrs_t in reversed(backpointers):
        best_tag_id = bptrs_t[best_tag_id]
        best_path.append(best_tag_id)
        # pop off the start tag
        # we don't want to return that ti the caller
    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG] # Sanity check
    best_path.reverse() # 把从后向前的路径正过来
    return path_score, best_path
```

