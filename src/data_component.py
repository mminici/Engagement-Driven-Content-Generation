import networkx as nx
import numpy as np
import pickle
from pathlib import Path
import os

from synthetic_generator import generate_G_and_opinions
from utils import build_retweet_exposure_graph

from operator import itemgetter


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
            self.G, self.opinions, self._node2community = load_referendum_dataset(exposure_graph=False, community="")
        elif real_data == "Referendum-exp":
            self.G, self.opinions, self._node2community = load_referendum_dataset(exposure_graph=True, community="")
        elif real_data == "Brexit":
            self.G, self.opinions, self._node2community = load_brexit_dataset(exposure_graph=False, community="")
        elif real_data == "Brexit-positive":
            self.G, self.opinions, self._node2community = load_brexit_dataset(exposure_graph=False, community="positive")
        elif real_data == "Brexit-negative":
            self.G, self.opinions, self._node2community = load_brexit_dataset(exposure_graph=False, community="negative")
        elif real_data == "Brexit-exp":
            self.G, self.opinions, self._node2community = load_brexit_dataset(exposure_graph=True, community="")
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
        return self.G.number_of_nodes()

    def get_graph(self):
        return self.G.copy()

    def get_opinions(self):
        return np.copy(self.opinions[list(self.G.nodes())])
        # return np.copy(itemgetter(*set(self.G.nodes()))(self.opinions))

    def get_opinion_mean(self):
        return np.mean(self.opinions[list(self.G.nodes())])
        # return np.mean(itemgetter(*set(self.G.nodes()))(self.opinions))

    def get_neighbors(self, node_id):
        if self.neighbors is None:
            return list(nx.neighbors(self.G, node_id))
        return self.neighbors[node_id]

    def get_opinion(self, node_id):
        return self.opinions[node_id]

    def update_opinion(self, node_id, new_opinion):
        self.opinions[node_id] = new_opinion
        return self.opinions[node_id]

    #def update_opinions(self, new_opinion_vector):
    #    self.opinions = new_opinion_vector
    #    return self.opinions

    def pre_compute_neighboring(self):
        neighbors_dict = {}
        for node_id in self.G.nodes():
            neighbors_dict[node_id] = self.get_neighbors(node_id)
        self.neighbors = neighbors_dict


load_referendum_dataset = lambda exposure_graph:  _load_real_data(base_folder=Path("/mnt/nas/cinus/SocialAIGym/data/raw/Referendum"), hashing="ita_referendum_04", exposure_graph=exposure_graph)
load_brexit_dataset = lambda exposure_graph, community:  _load_real_data(base_folder=Path("/mnt/nas/cinus/SocialAIGym/data/raw/Brexit"), hashing="brexit_07", exposure_graph=exposure_graph, community=community)


def _load_real_data(base_folder: Path, hashing: str, exposure_graph: bool=False, reverse: bool=False, community: str=""):
    """Loads dataset from base_folder folder
    """
    # A. Load directed graph
    if not exposure_graph:
        print("Loading follow graph ..")
        g = nx.read_edgelist(base_folder / Path(f"{hashing}_edgelist.txt"), nodetype=int, create_using=nx.DiGraph)
        g_community = None
        base_community_path = "/home/coppolillo/Desktop/LLMs/SocialAIGym/data/processed/"
        
        if community == "positive" or community == "negative": # community 0 or 13 of the reduced graph
            print(f"Retrieving the {community} community from the reduced graph")
            g_community = nx.read_edgelist(os.path.join(base_community_path, f"{community}_community_graph_reduced.npy"), 
                                 create_using=nx.DiGraph, edgetype=int, nodetype=int)
            
        if reverse:
            print("Reverse edge directionality! \n BEFORE: u->v: u follows v \n NOW u<-v: propagation goes from v to u  )")
            g = nx.DiGraph.reverse(g)
    else:
        print("Building exposure graph ..")
        o_folder = base_folder / Path(f"{hashing}_propagations_and_polarities.pkl")
        with open(o_folder, "rb") as f:
            propagations, _ = pickle.load(f)
        g = build_retweet_exposure_graph(propagations)
    
    print("Graph loaded  ✅")
    if g_community is None:
        _node_label_types = {type(node).__name__ for node in g.nodes()}
        print(f"|V|={g.number_of_nodes():_} |E|={g.number_of_edges():_} node types={_node_label_types}")
    else:
        _node_label_types = {type(node).__name__ for node in g_community.nodes()}
        print(f"|V|={g_community.number_of_nodes():_} |E|={g_community.number_of_edges():_} node types={_node_label_types}")
    
    # B. Opinions from stance. Cast to [0, 1] range
    with open(base_folder / Path(f"{hashing}_node2stance.pkl"), "rb") as f_handle:
        node2stance = pickle.load(f_handle)
        
    def normalize_array(arr):
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        normalized_arr = (arr - arr_min) / (arr_max - arr_min)
        return normalized_arr

    opinions = normalize_array(np.array([node2stance[i] for i in range(g.number_of_nodes())]))
    # opinions_value = normalize_array(np.array([node2stance[i] for i in range(g.number_of_nodes())]))

    # now opinions is a dict so it is not needed that len(opinions) == g.number_of_nodes()
    # opinions = dict(zip(list(range(g.number_of_nodes())), opinions_value))

    # NOT USED
    
    # C. Assign 2 communities in the undirected graph using 'kernighan_lin_bisection'
    communities = nx.algorithms.community.kernighan_lin_bisection(g.to_undirected())
    _node2community = np.zeros(g.number_of_nodes())
    # _node2community[list(communities[1])] = 1

    if g_community is not None:
        return g_community, opinions, _node2community
    
    return g, opinions, _node2community