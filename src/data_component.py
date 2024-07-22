import networkx as nx
import numpy as np
import pickle
from pathlib import Path

from synthetic_generator import generate_G_and_opinions


class DataComponent:
    def __init__(self, 
                 num_nodes=None, 
                 modularity=None, homophily=None, 
                 avg_deg=None, 
                 alpha=None, beta=None, 
                 real_data:str=None):
        self.neighbors = None
        self.num_nodes = num_nodes
        self.modularity = modularity
        self.homophily = homophily
        if real_data == "Referendum":
            self.G, self.opinions, self._node2community = load_referendum_dataset()
        elif real_data == "Brexit":
            self.G, self.opinions, self._node2community = load_brexit_dataset()
        elif real_data is None:
            self.G, self.opinions, self._node2community = generate_G_and_opinions(N=num_nodes,
                                                            avg_deg=avg_deg,
                                                            mu=1-modularity, # mu := Fraction of inter-community edges incident to each node
                                                            conformism=homophily,
                                                            alpha=alpha,
                                                            beta=beta)
        else:
            raise Exception(f"real_data={real_data} not in (Referendum, Brexit, None)")
        
            
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


load_referendum_dataset = lambda:  _load_real_data(base_folder=Path("/mnt/nas/cinus/SocialAIGym/data/raw/Referendum"), hashing="ita_referendum_04")
load_brexit_dataset = lambda:  _load_real_data(base_folder=Path("/mnt/nas/cinus/SocialAIGym/data/raw/Brexit"), hashing="brexit_07")


def _load_real_data(base_folder: Path, hashing: str):
    """Loads dataset from base_folder folder
    """
    # A. Load directed graph
    g = nx.read_edgelist(base_folder / Path(f"{hashing}_edgelist.txt"), nodetype=int, create_using=nx.DiGraph)

    # B. Opinions from stance. Cast to [0, 1] range
    with open(base_folder / Path(f"{hashing}_node2stance.pkl"), "rb") as f_handle:
        node2stance = pickle.load(f_handle)
    def normalize_array(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        normalized_arr = (arr - arr_min) / (arr_max - arr_min)
        return normalized_arr

    opinions = normalize_array(np.array([node2stance[i] for i in range(g.number_of_nodes())]))

    # C. Assign 2 communities in the undirected graph using 'kernighan_lin_bisection'
    communities = nx.algorithms.community.kernighan_lin_bisection(g.to_undirected())
    _node2community = np.zeros(g.number_of_nodes())
    _node2community[list(communities[1])] = 1
    
    return g, opinions, _node2community