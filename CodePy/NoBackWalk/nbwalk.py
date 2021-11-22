from genwalk import WalkGenerator
from gensim.models import Word2Vec
from scipy.stats import bernoulli
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def visualize(G):
    # 可视化
    pos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, group_1, label = True,
    node_size = 50, node_color = 'r')
    nx.draw_networkx_nodes(G, pos, group_2, node_size = 50,
                                node_color = 'b')
    nx.draw_networkx_nodes(G, pos, group_3, node_size = 50,
                                node_color = 'k')

    nx.draw_networkx_edges(G, pos, label= True, alpha = 0.5)

    plt.show()      
 
def get_embeddings(model):
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[node]

    return embeddings

def train(walks, embed_size, window_size, workers, epochs, **kwargs):
    kwargs["sentences"] = walks
    kwargs["min_count"] = kwargs.get("min_count", 0)
    kwargs["vector_size"] = embed_size
    kwargs["sg"] = 1
    kwargs["hs"] = 0  # not use Hierarchical Softmax
    kwargs["workers"] = workers
    kwargs["window"] = window_size
    kwargs["epochs"] = epochs

    print("Learning embedding vectors...")
    model = Word2Vec(**kwargs)
    print("Learning embedding vectors done!")

    return model


if __name__ == "__main__":
    block_matrix = np.array([[0.8,0.05,0.05],
                            [0.05,0.8,0.05],
                            [0.05, 0.05, 0.8]])
                            
    num_nodes = 90
    K = 3    # 社团个数

    # 为每个结点分配属于的社团
    # 前25个结点属于社团1,中间30个结点属于社团2,后面35个结点属于社团3
    group_1 = list(range(25))
    group_2 = list(range(25,55))
    group_3 = list(range(55,90))

    group = [group_1, group_2, group_3]

    adj_matrix =  np.zeros((num_nodes, num_nodes), dtype=np.int32)


    # 生成无向图邻接矩阵
    for i, group_i in enumerate(group):
        for j, group_j in enumerate(group):
            if (j >= i):
                for node_i in group_i:
                    for node_j in group_j:
                        adj_matrix[node_i][node_j] = bernoulli.rvs(block_matrix[i][j], random_state=None)

    # 再令对角线上元素为0, 无自环
    for i in range(num_nodes):
        adj_matrix[i][i] = 0

    G = nx.from_numpy_array(adj_matrix)

    walker = WalkGenerator(G)
    walks = walker.simulate_walks(num_walks=80, walk_length=10)   # 生成随机游走训练集
    model = train(walks, embed_size=2, window_size=5, workers=3, epochs=5) # 训练
    embeddings = get_embeddings(model)  # 得到结点embeddings
    print(embeddings)
