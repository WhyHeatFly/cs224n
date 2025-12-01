## *a2_question answer*
### *1. understanding_word2vec*  
(a): 由于 $y$ 是真实标签，因此对于 $w ∈ Vocab$, 有:  
$\quad$ $\quad$ $\quad$   $y_w = 0,\quad w \neq o$  
$\quad$ $\quad$ $\quad$   $y_w = 1, \quad w = o $  
$\quad$ $\quad$ $\quad$ $∴ -\sum_{w∈Vocab} \quad y_wlog(\hat{y}_w) = -log(\hat{y}_o)$  

(b): 

i. 求 $\frac{\partial J(v_c, o, U)}{\partial v_c}$ , 其中  $J=-log(\hat{y_o})$, $\hat{y_w}=\frac{exp(u_w^Tv_c)}{\sum_{x}exp(u_x^Tv_c)}$ 

$∴ \frac{\partial J}{\partial v_c}= \frac{\partial -log(\hat{y_o})}{\partial v_c} = \frac{\partial [-log(\frac{exp(u_o^Tv_c)}{\sum_{x}exp(u_x^Tv_c)})]}{\partial v_c} \\=  \frac{\partial [-log(exp(u_o^Tv_c)) + log(\sum_{x}exp(u_x^Tv_c))]}{\partial v_c} = -\frac{\partial log(exp(u_o^Tv_c))}{\partial v_c} + \frac{\partial log(\sum_{x}exp(u_x^Tv_c))}{\partial v_c}\\= - \frac {\partial u_o^Tv_c}{\partial v_c} + \frac {\partial log(\sum_{x}exp(u_x^Tv_c))}{\partial v_c} = -u_o + \frac{1}{\sum_xexp(u_x^Tv_c)} ·\sum_x exp(u_x^Tv_c) · v_c \\ = - u_o + \sum_x \hat{y}_x v_x = U^T(\hat{y} - y)$ 

 其中$U$的每一行对应一个$U_w^T$   

ii. 当$\hat{y} = y$ 时，即预测分布恰好等于真实分布时，梯度为0  

iii.  

(c). $L_2$ 正则化： $U_{norm} = \frac {U}{||U||_2} = \frac {U}{\sqrt{\sum_i u_i^2}}$ 会将向量长度归一化。故当向量长度对于预测标签的$label$存在影响时，归一化会破坏有效信息 

(d). $\frac {\partial J}{\partial U} = [\frac {\partial J}{\partial u_1}, \frac {\partial J}{\partial u_2}, ... , \frac {\partial J}{\partial u_|vocab|}]^T$ 

(e). $\frac {\partial J}{\partial U} = (\hat{y_w} - y_w)·v_c\quad$  具体的：  
若$w=o: \frac {\partial J}{\partial u_o} = (\hat{y_o} - 1) v_c \\$  若$w \neq o: \frac {\partial J}{\partial u_w} = \hat{y_w}v_c$  

### *2. Machine Learning & Neural Networks*  
(a):  
i. Momentum 通过在更新中加入过去梯度的指数平均，使参数更新方向更加平滑，而不是被每个 mini-batch 的噪声梯度左右。这样可以减少梯度在不同方向间“来回震荡”，特别是在损失函数狭长谷地中能够更快向正确方向移动。较低的更新方差能让训练更加稳定，并能加速收敛到更好的解  

ii. 因为Adam用（梯度平方的指数平均）来缩放更新量，所以梯度一直很大的参数会被缩小更新，而梯度 consistently 很小的参数会被放大更新。这让模型可以自动调节不同参数的学习率：对变化缓慢的参数给更大更新，对震荡剧烈的参数给更小更新。这样既能避免某些参数步子过大导致训练不稳定，也能让那些学习缓慢的参数更快收敛。  
(b):  
i. 

