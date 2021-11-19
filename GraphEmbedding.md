1. 大体介绍4种算法，以及注意事项



基于近邻相似假设

| Algorithm |                      Method                       |           Model            |
| :-------: | :-----------------------------------------------: | :------------------------: |
| DeepWalk  |                   无偏随机游走                    |      Skip-Gram model       |
| Node2vec  |                   有偏随机游走                    |      Skip-Gram model       |
|   LINE    | 一阶相似性：邻接结点对 \| 二阶相似性：结点1阶邻居 | KL-divergence optimization |
|   SDNE    |   一阶相似性：邻接结点对\| 二阶相似性：邻接矩阵   | semi-supervised deep model |

因为社团内部有很多边相连，社团与社团之间连边较少，随机游走偏向于在社团内部进行游走，而我们的模型中通常会从同一结点进行多次随机游走。

随机游走起始结点选择为

* 社团连接结点：则其大概率会进入其中一个社团进行随机游走。
* 社团内部结点：大概率在社团内部进行随机游走。

因此，相似的结点或同属于同一个社团的结点在嵌入空间中的距离更为接近，可以通过简单的将两个向量做内积然后对结果进行排序。

LINE、SDNE都是重构一阶相似性和二阶相似性，只是使用的模型不一样。



参考资料：

[CS224W | Home (stanford.edu)](http://web.stanford.edu/class/cs224w/)

[Graph Representation Learning Book (mcgill.ca)](https://www.cs.mcgill.ca/~wlh/grl_book/)

[深度学习中不得不学的Graph Embedding方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/64200072)

[【Graph Embedding】DeepWalk：算法原理，实现和应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/56380812)

