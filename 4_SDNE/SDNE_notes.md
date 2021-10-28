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
于是对任意的向量$\boldsymbol{\mathrm{y}_i}=[y_{i}^{(1)},...,y_{i}^{(k)}, ...,y_{i}^{(d)}]^T$. 显然其为$Y$的列向量， $Y=\{\boldsymbol{\mathrm{y}_i}\}_{i=1}^{n}$ . 令$\boldsymbol{\mathrm{y}}^{(k)} \in R^n$为矩阵$Y$的行向量.
$$
\begin{aligned}
\mathcal{L}_{1st}=\sum_{i,j=1}^n s_{ij} ||\boldsymbol{\mathrm{y}_i}-\boldsymbol{\mathrm{y}_j}||_2^2&=
\sum_{i,j=1}^n s_{ij} \sum_{k=1}^d (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= \sum_{k=1}^d \sum_{i,j=1}^n s_{ij} (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= 2\sum_{k=1}^d \boldsymbol{\mathrm{y}}^{(k)T} L \boldsymbol{\mathrm{y}}^{(k)} \\
&= 2tr(Y^TLY)
\end{aligned}
$$

Left work 度量的理解



# 拉普拉斯矩阵

## 定义

无向图$G=(V,E)$，$A \in R^{n \times n}$为邻接矩阵，其元素
$$
a_{ij}=\begin{cases}
1 & \mathrm{if}\ (v_i,v_j) \in E \\
0 & \mathrm{else}
\end{cases}
$$
$N(i)$为结点$v_i$的邻居，$D \in R^{n \times n}$为度矩阵，对角矩阵，其元素
$$
d_{ii}= \sum_{j=1}^n a_{ij}= \sum _{j \in N(i)} a_{ij}
$$
定义拉普拉斯矩阵(Laplacian matrix) $L=D-A$，其元素
$$
l_{ij}=
\begin{cases}
d_i & \mathrm{if}\ i=j \\
-1 & \mathrm{if}\ (v_i,v_j) \in E  \\
0 & \mathrm{otherwise}
\end{cases}
$$
正则化表达形式(symmetric normalized laplacian) $L_{\mathrm{sym}}=D^{-1/2}LD^{-1/2}$，其元素
$$
l_{\mathrm{sym}}(i,j)=
\begin{cases}
1 & \mathrm{if}\ i=j \\
\frac{-1}{\sqrt{d_i d_j}}  & \mathrm{if}\ (v_i,v_j) \in E \\
0 & \mathrm{otherwise}
\end{cases}
$$
定义向量$\boldsymbol{x}=[x_1,x_2,···,x_n]^\top$，则
$$
\begin{aligned}
L\boldsymbol{x}=(D-A)\boldsymbol{x}&=[···, d_ix_i-\sum_{j }a_{ij}x_j,···]^T
\\
&= [···,\sum _{j } a_{ij} x_i - \sum _{j } a_{ij} x_j,···]^T \\
&= [···, \sum _{j }a_{ij}(x_i-x_j),···]^T
\end{aligned}
$$
由此可知，拉普拉斯矩阵是反映向量局部平滑度的算子。接着我们定义二次型
$$
\begin{aligned}
\boldsymbol{x}^TL\boldsymbol{x}&=\sum_i x_i \sum _{j }a_{ij}(x_i-x_j) \\
&= \sum_{ij} a_{ij}(x_i^2-x_ix_j)
\end{aligned}
$$
调换$i,j$位置，我们得到
$$
\boldsymbol{x}^TL\boldsymbol{x}=\sum_{ij} a_{ij}(x_i^2-x_ix_j)=\sum_{ij} a_{ij}(x_j^2-x_ix_j)
$$
将右边等式两边相加，于是
$$
\begin{aligned}
\boldsymbol{x}^TL\boldsymbol{x}&=\frac{1}{2} \sum_{ij} a_{ij}(x_i^2-2x_ix_j+x_j^2) \\
&= \frac{1}{2}\sum_{ij} a_{ij}(x_i-x_j)^2

\end{aligned}
$$

## 来源

拉普拉斯矩阵的定义来源于拉普拉斯算子，$n$维欧式空间中的一个二阶微分算子：$\Delta f=\sum_{j=1}^n \frac{\partial ^2 f}{\partial x_i^2}$。将该算子退化到离散二维图像空间就是边缘检测算子：
$$
\begin{aligned}
\Delta f(x,y) &=
\frac{\partial ^2 f(x,y)}{\partial x^2} + 
\frac{\partial ^2 f(x,y)}{\partial y^2}\\
&= [(f(x+1,y)-f(x,y))-(f(x,y)-f(x-1,y))]\\
&+ [(f(x,y+1)-f(x,y))-(f(x,y)-f(x,y-1))]\\
&= [f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)] -4f(x,y)
\end{aligned}
$$
图像处理中通常被当作模板的形式：
$$
\begin{bmatrix} 0 & 1 & 0\\ 1 & -4 & 1 \\0 & 1 & 0 \end{bmatrix}
$$
拉普拉斯算子用来描述中心像素与局部上、下、左、右四邻居像素的差异，这种**性质**经常也被用来当作图像上的边缘检测算子。**Leave**

拉普拉斯矩阵反映图信号**局部平滑度**的算子。

实对称矩阵，可被单位对角化。

通过拉普拉斯矩阵可以定义一个二次型，称为总变差，刻画图信号**整体平滑度**。

