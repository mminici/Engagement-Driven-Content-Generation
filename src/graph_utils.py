import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_pos_from_communities(node2community, g, seed=42):
    import collections
    np.random.seed(seed)
    numb_comunities = max(node2community) + 1
    community2count = collections.Counter(node2community)
    A = np.zeros((numb_comunities, numb_comunities))
    for u in range(numb_comunities):
        for v in range(numb_comunities):
            A[u, v] = 100 / (5*(community2count[u] + community2count[v]))
    G_centroids = nx.from_numpy_matrix(A)
    pos_centroids = nx.spring_layout(G_centroids, weight='weight', seed=seed)

    pos = {}
    for centroid in G_centroids.nodes():
        centr_pos = pos_centroids[centroid]
        nodes_in_community = np.where(node2community == centroid)[0]
        for i, node in enumerate(sorted(nodes_in_community, key=lambda x: g.degree[x], reverse=True)):
            pos[node] = np.array([2,2])
            pos[node] = centr_pos + .2*i*np.random.uniform(low=-.025, high=.025, size=2)+np.array([np.random.choice([-1, 1])*.1, np.random.choice([-1, 1])*.1])
    return pos


def plot_graph_config(data, llm_node_id, filename=None):
    fig, ax = plt.subplots(1,1, figsize=(7, 5))
    
    pos = create_pos_from_communities(data._node2community, data.G, seed=42)
    cmap = plt.colormaps.get_cmap('coolwarm')
    norm = plt.Normalize(vmin=0, vmax=1)
    node_colors = cmap(norm(data.opinions))
    nx.draw_networkx_edges(data.G, pos, alpha=0.05, ax=ax)

    # Draw all nodes except the special node
    non_special_nodes = [node for node in data.G.nodes if node != llm_node_id]
    nx.draw_networkx_nodes(data.G, pos, nodelist=non_special_nodes, node_color=node_colors[1:], cmap=cmap, ax=ax)
    # Draw the special node
    nx.draw_networkx_nodes(data.G, pos, nodelist=[llm_node_id], node_color=np.array([[0, 1, 0, 1]]), node_size=750, ax=ax)
    # Draw labels if needed
    #nx.draw_networkx_labels(data.G, pos)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    # Display the graph
    plt.title(f"Opinions: avg={np.mean(data.opinions):.2f}, std={np.std(data.opinions):.2f}, min={np.min(data.opinions):.2f}, max={np.max(data.opinions):.2f}")
    
    if filename:
        
        plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def LLM_central(data, measure="betweenness"):
    betweenness_centrality = nx.betweenness_centrality(data.get_graph())
    sorted_by_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
    llm_node_id = sorted_by_betweenness[0][0]
    return llm_node_id


def LLM_in_comm(data, size: str):
    """Returns index of LLM in a community
        size = largest --> returns LLM in community with max size
        size = smallest --> returns LLM in community with min size
    """
    from collections import defaultdict
    community_dict = defaultdict(list)
    for node, community in enumerate(data._node2community):
        community_dict[community].append(node)
    if size == "largest":
        community = max(community_dict.items(), key=lambda x: len(x[1]))[1]
    elif size == "smallest":
        community = min(community_dict.items(), key=lambda x: len(x[1]))[1]
    else:
        raise Exception(f"size={size} not in (largest, smallest)")
    return community[0]

def LLM_in_echochamber(data, opinion_position: str):
    """Returns index of LLM in an echo chamber
        opinion_position = high --> returns LLM in community with max average opinion
        opinion_position = low --> returns LLM in community with min average opinion
    """
    from collections import defaultdict
    # Group nodes by their community assignments
    community_dict = defaultdict(list)
    for node, community in enumerate(data._node2community):
        community_dict[community].append(node)

    # Calculate the average opinion for each community
    community_avg_opinion = {community: np.mean(data.opinions[nodes]) for community, nodes in community_dict.items()}

    # Find the community with the highest average opinion
    if opinion_position == "high":
        target_community = max(community_avg_opinion, key=community_avg_opinion.get)
    elif opinion_position == "low":
        target_community = min(community_avg_opinion, key=community_avg_opinion.get)
    else:
        raise Exception(f"opinion_position={opinion_position} not in (high, low)")

    # Select a node from the target community
    return community_dict[target_community][0]