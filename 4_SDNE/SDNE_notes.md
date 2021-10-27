图嵌入的三个挑战

* 高度非线性 High non-linearity

    

* 结构保持 Structure-Preserving

    网络嵌入要能够保持原来网络中的结构

* 稀疏性 Sparsity

    现实网络中的边非常稀疏，只能利用有限的可见边（limited observed links），并不足以达到满意的表现

对于高度非线性，利用深度学习模型

对于结构保持和稀疏性，提出1阶和2阶相似性加入到训练过程

1阶相似性：针对有边相连的两个结点，local pairwise similarity, local network structure. 但由于网络的稀疏性，一些合理的边丢失了（legitimate links），1阶相似性不足够表示网络的结构。

2阶相似性：两个结点邻居结构的相似性，global network structure.

如何加入到训练过程？

半监督学习框架：

* 无监督部分重构2阶相似性，去保持global nework structure.
* 有监督部分引入1阶相似性，保持local network structure.

由于稀疏网络中具有2阶相似性的结点对的数量更多，所以能够提供更多有效的信息，2阶相似性的引入缓解了稀疏性。

# 公式推导

对任意向量$\boldsymbol{\mathrm{y}}$，邻接矩阵$S$，拉普拉斯矩阵 $L=D-S$，$D$是对角矩阵 $d_i=\sum_j s_{ij}$.
$$
\begin{aligned}
\boldsymbol{\mathrm{y}}^TL\boldsymbol{\mathrm{y}}&= 
\boldsymbol{\mathrm{y}}^TD\boldsymbol{\mathrm{y}}-\boldsymbol{\mathrm{y}}^TS\boldsymbol{\mathrm{y}}\\
&= \sum_i d_iy_i^2-\sum_{i,j}s_{ij}y_iy_j\\
&= \frac{1}{2}(\sum_i d_iy_i^2  -2\sum_{i,j}s_{ij}y_iy_j + \sum_j d_jy_j^2) \\
&=\frac{1}{2}(\sum_{ij} s_{ij}y_i^2  -2\sum_{i,j}s_{ij}y_iy_j + \sum_{ij} s_{ij}y_j^2) \\
&= \frac{1}{2}\sum_{ij} s_{ij}(y_i-y_j)^2
\end{aligned}
$$
于是对任意的向量$\boldsymbol{\mathrm{y}_i}=[y_{i}^{(1)},...,y_{i}^{(k)}, ...,y_{i}^{(d)}]^T$. 显然其为$Y$的列向量， $Y=\{\boldsymbol{\mathrm{y}_i}\}_{i=1}^{n}$ . 令$\boldsymbol{\mathrm{y}}^{(k)}$为矩阵$Y$的行向量.
$$
\begin{aligned}
\mathcal{L}_{1st}=\sum_{i,j=1}^n s_{ij} ||\boldsymbol{\mathrm{y}_i}-\boldsymbol{\mathrm{y}_j}||_2^2&=
\sum_{i,j=1}^n s_{ij} \sum_{k=1}^d (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= \sum_{k=1}^d \sum_{i,j=1}^n  (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= 2\sum_{k=1}^d \boldsymbol{\mathrm{y}}^{(k)T} L \boldsymbol{\mathrm{y}}^{(k)} \\
&= 2tr(Y^TLY)
\end{aligned}
$$

Left work 度量的理解
