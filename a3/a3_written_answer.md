## a3_written_answer
###  1. Neural Machine Translation with RNNs  
(g). enc_mask使所有的pad位置都初始化为负无穷，这样在计算attention_score的时候，得到的分数会约等于0，可以更好的让模型把注意力放在有真正含义的语句上。

(h).   

(i).  
i. dot product vs multiplicative  
没有额外参数：不用学一个 𝑊, 结构更简单；  
计算非常快：就是纯矩阵乘法，GPU 上效率极高；  
要求 𝑠𝑡 和 ℎ𝑖 维度必须相同；  

表达力有限：只是“角度 + 长度”的相似度，没有可学习的变换；  
在维度很大时，向量内积的值会很大，经softmax变得很尖锐（Transformer 里才引入 scaled dot-product，把它除以 $\sqrt{d}$）。  
ii. additive vs multiplicative：  
优点：additive attention 通过 𝑊1,𝑊2和 tanh 形成一个小 MLP，表达能力更强，可以更灵活地建模 𝑠𝑡 和 ℎ𝑖 之间的关系，特别是在维度较低时效果更好。  

缺点：每一步都要做较重的前馈计算，比 multiplicative 更慢、参数更多，在大模型 / 长序列上代价更高。

### 2. Analyzing NMT Systems  
(a).通过在decoder前加入conv1d，可以让模型学习每个字词附近的字词，进而更好的理解原文