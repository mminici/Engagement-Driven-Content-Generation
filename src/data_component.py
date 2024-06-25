import networkx as nx
import numpy as np

from synthetic_generator import generate_G_and_opinions


class DataComponent:
    def __init__(self, num_nodes, modularity, homophily, avg_deg, alpha, beta):
        self.neighbors = None
        self.num_nodes = num_nodes
        self.modularity = modularity
        self.homophily = homophily
        self.G, self.opinions, self._node2community = generate_G_and_opinions(N=num_nodes,
                                                           avg_deg=avg_deg,
                                                           mu=modularity,
                                                           conformism=homophily,
                                                           alpha=alpha,
                                                           beta=beta)

    def get_num_nodes(self):
        return len(self.opinions)

    def get_graph(self):
        return self.G.copy()

    def get_opinions(self):
        return np.copy(self.opinions)

    def get_opinion_mean(self):
        return np.mean(self.opinions)

    def get_neighbors(self, node_id):
        if self.neighbors is None:
            return list(nx.neighbors(self.G, node_id))
        return self.neighbors[node_id]

    def get_opinion(self, node_id):
        return self.opinions[node_id]

    def update_opinion(self, node_id, new_opinion):
        self.opinions[node_id] = new_opinion
        return self.opinions[node_id]

    def update_opinions(self, new_opinion_vector):
        self.opinions = new_opinion_vector
        return self.opinions

    def pre_compute_neighboring(self):
        neighbors_dict = {}
        for node_id in self.G.nodes():
            neighbors_dict[node_id] = self.get_neighbors(node_id)
        self.neighbors = neighbors_dict