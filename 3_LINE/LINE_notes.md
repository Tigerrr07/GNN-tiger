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



# 负采样



真实logistic function加上噪声分布采集的$k$个sample的logistic function. Replace every $\mathrm{log}P(w_O|w_I)$ term by the objective:
$$
\mathrm{log}\ \sigma(v_{wO}'^\top \ v_{w_I})+
$$
