import networkx as nx
import numpy as np

from utils import Timeout

DEFAULT_AVG_DEG = 12


def fortunato_benchmark(n, avg_deg=DEFAULT_AVG_DEG, mu=0.1, seed=0, verbose=False, power_law_coef=2.):
    G = None
    MAX_LIM = 10
    while G is None and MAX_LIM > 0:
        def _compute_G():
            return nx.generators.community.LFR_benchmark_graph(n, tau1=power_law_coef, tau2=1.1, mu=mu,
                                                               average_degree=avg_deg, min_community=(n // 20),
                                                               max_iters=50, seed=seed).to_directed()
        try:
            with Timeout(15):
                G = _compute_G()
        except nx.ExceededMaxIterations as e:
            if verbose:
                print(e, '\tRetrying...')
            seed += 150
            MAX_LIM -= 1
        except Timeout.Timeout:
            if verbose:
                print("Timeout \tRetrying...")
            seed += 1
            MAX_LIM -= 1
    if G is None:
        return G, None
    if verbose:
        print("Obtained Average Degree: %.3f in, %.3f out" %
              (np.mean(list(dict(G.in_degree).values())),
               np.mean(list(dict(G.out_degree).values())))
              )
    communities = {tuple(sorted(G.nodes[i]['community'])) for i in G.nodes}
    communities = [set(c) for c in communities]
    node2community = np.array([next(i for i, comm in enumerate(communities) if node in comm) for node in G.nodes])

    return G, node2community


def assign_opinions(G, node2community, centrism=1, conformism=0.5, distr="beta", innovators_perc=1):
    if distr == "beta":
        community_opinions = np.random.beta(centrism, centrism, size=(max(node2community) + 1))
        opinions = np.random.beta(centrism, centrism, size=G.number_of_nodes())
    elif distr == "uniform":
        number_of_innovators = int(G.number_of_nodes() * innovators_perc)
        innovators_opinions = np.random.uniform(0.5, 1., number_of_innovators)
        contrarians_opinions = np.random.uniform(0., 0.5, G.number_of_nodes() - number_of_innovators)
        opinions = np.concatenate([innovators_opinions, contrarians_opinions])
        opinions = np.random.permutation(opinions)
        number_of_innovators_community = int((max(node2community) + 1) * innovators_perc)
        community_innovators_opinions = np.random.uniform(0.5, 1., number_of_innovators_community)
        community_contrarians_opinions = np.random.uniform(0., 0.5,
                                                           (max(node2community) + 1) - number_of_innovators_community)
        community_opinions = np.concatenate([community_innovators_opinions, community_contrarians_opinions])
        community_opinions = np.random.permutation(community_opinions)
    else:
        print(f"bad choice: {distr}")
        return
    for node in range(G.number_of_nodes()):
        if np.random.random() < conformism:
            opinions[node] = community_opinions[node2community[node]]
    return opinions


def generate_G_and_opinions(N, mu=.1, centrism=1., conformism=0.9, avg_deg=DEFAULT_AVG_DEG, distr="beta", use_lcc=True,
                            seed=12121995, verbose=False, innovators_perc=.5, power_law_coef=2.):
    G, node2community = fortunato_benchmark(N, avg_deg=avg_deg, mu=mu, seed=seed, verbose=verbose,
                                            power_law_coef=power_law_coef)
    if G is None:
        return None, None, None
    opinions = assign_opinions(G, node2community, centrism=centrism, conformism=conformism, distr=distr,
                               innovators_perc=innovators_perc)
    if use_lcc:
        # extracting largest connected component
        giant = max(nx.connected_components(nx.Graph(G)), key=len)
        lcc = nx.subgraph(G, giant)
        # remapping nodes into 0-N range
        all_nodes = list(lcc.nodes())
        G = nx.relabel_nodes(lcc, {all_nodes[i]: i for i in range(len(all_nodes))})
        opinions = opinions[list(giant)]
        node2community = node2community[list(giant)]
        # remap community IDs
        communities_left_after_extracting_lcc = list(set(node2community))
        new_mapping = {communities_left_after_extracting_lcc[i]: i for i in
                       range(len(communities_left_after_extracting_lcc))}
        node2community = np.array(list(map(lambda x: new_mapping[x], node2community)))
    return G, opinions, node2community
