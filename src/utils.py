import signal
import networkx as nx


class Timeout:
    """Timeout class using ALARM signal"""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def generate_time_ordered_pairs(prop):
    """Function to generate time-ordered pairs from one propagation list
       prop: Iterable : ordered list of active users (first to last in time)
       
       Returns pairs u->v (u activated before v, so u can influence v) for a given prop list
    """
    pairs = []
    for i in range(len(prop)):
        for j in range(i+1, len(prop)):
            pairs.append((int(prop[i]), int(prop[j])))
    return pairs

def build_retweet_exposure_graph(propagations):
    """propagations: Iterable All propagations are list ordered by time (first active, to last active user)
       Returns influence graph given by exposure to previous content: 
       u --> v: u can influence v since it activated before v in a propagation.
    """
    prop_edge_list = [generate_time_ordered_pairs(prop) for prop in propagations]

    # Flatten the list of lists of pairs to get a single list of pairs
    all_pairs = [pair for sublist in prop_edge_list for pair in sublist]

    # Fill the graph
    G = nx.DiGraph()
    for pair in all_pairs:
        G.add_edge(*pair) 
    return G
