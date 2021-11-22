import random

class WalkGenerator:
    def __init__(self, graph):  # G是networkx的graph
        self.graph = graph

    def noback_walk(self, walk_length, start_node):
        G = self.graph
        # 默认无向图是连通的, 从任一结点开始出发做RW, 必然能选取邻居结点
        walk = [start_node]   # 添加源结点
        cur_nbrs = list(G.neighbors(start_node))
        walk.append(random.choice(cur_nbrs))  # 添加源结点邻居结点


        while len(walk) < walk_length:
            cur_node = walk[-1]
            last_node = walk[-2]

            cur_nbrs = list(G.neighbors(cur_node))  # cur_nbrs必有last_node

            if len(cur_nbrs) == 1:  # 度为1, cur_nbrs==last_node, 不能继续游走
                break
            else:
                prob_vex = random.choice(cur_nbrs)
                while prob_vex == last_node:
                    prob_vex = random.choice(cur_nbrs)
                walk.append(prob_vex)
                    
        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.graph
        nodes = list(G.nodes())
        walks = []

        for i in range(num_walks):
            random.shuffle(nodes)
            for start_node in nodes:
                walks.append(self.noback_walk(walk_length, start_node))
        
        return walks