

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