# 交叉熵

概率分布$q$，真实分布$p$，
$$
\begin{aligned}
H(p,q)&=\mathbb{E}_p(-\mathrm{log}\ q(x)) \\
&= -\sum_x p(x)\ \mathrm{log}\ q(x)
\end{aligned}
$$

# KL散度

KL散度（KL Divergence），也叫KL距离或相对熵，是用概率分布$q$来近似$p$所造成的信息损失量。
$$
\begin{aligned}
KL(p,q)&=H(p,q)-H(p)\\
&= \sum_x p(x) \mathrm{log} \frac{p(x)}{q(x)}
\end{aligned}
$$



Present several extesions of the original Skim-Gram model.

* Negative Sampling
    * faster training
    * better vector representation for frequent words, compared to hierarchical softmax
* Subsampling of frequent words
    * Significant speedup
    * Improve accuracy of the representation of less frequent words
* Word to phrase based model: treat the phrase as individual tokens during the training,

# Negative Sampling

真实logistic function加上噪声分布采集的$k$个sample的logistic function. Replace every $\mathrm{log}P(w_O|w_I)$ term by the objective:
$$
\mathrm{log}\ \sigma(v_{wO}'^\top \ v_{wI})+\sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\mathrm{log}(-v_{w}'^\top  v_{wI})]
$$
$P_n(w)$ is the noise distribution, 
$$
P(w_i)=\frac{f(w_i)^{3/4}}{\sum_j f(w_j)^{3/4}}
$$

# Subsampling of Frequent Words

Each word $w_i$ in the training set is **discareded** with probability computed by the formula
$$
P(w_i)=1-\sqrt{\frac{t}{f(w_i)}}
$$
$f(w_i)$ is the frequency of word $w_i$ and $t$ is  a chosen threshold, typically around $10^{-5}$. If $f(w_i) \leq t$，$P(w_i) \leq 0$. Subsampling words whose frequency is greater than $t$ while preserving the ranking of the frequencies. It work well in practice.



* 句法类比（Syntactic analogy）

    "quick" : "quickly" :: "slow" : "slowly"

* 语义类比（Semantic analogy）

    The country to capital city relationship.



# Learning Phrases

Many phrases have a meaning that is not a simple composition of the meaning of its individual words.
