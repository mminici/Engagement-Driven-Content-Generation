from abc import ABC, abstractmethod
import networkx as nx
from sklearn.preprocessing import normalize
import numpy as np



class OpinionDiffusionComponent(ABC):
    @abstractmethod
    def propagate_message_at_equilibrium(self, message, node_id):
        raise NotImplementedError()

    @abstractmethod
    def propagate_message_one_step(self, message, node_id):
        raise NotImplementedError()


class FJDiffusionComponent(OpinionDiffusionComponent):
    """Rebalancing Social Feed to Minimize Polarization and Disagreement
       Cinus et al, 
       https://dl.acm.org/doi/10.1145/3583780.3615025
    """
    def __init__(self, data_component):
        self.data_component = data_component
        self.n = self.data_component.get_num_nodes() # Numb nodes
        self.s = self.data_component.get_opinions().reshape((self.n, 1)) # Inner opinions
        self.z = self.data_component.get_opinions().reshape((self.n, 1)) # Expressed opinions at time T 
        self.A = nx.adjacency_matrix(self.data_component.G) # Adj
        self.A = normalize(self.A, axis=1, norm='l1') # Row stochastic Adj
        self.I = np.identity(self.n)
        self.equilibrium_mtx = np.linalg.inv(2 * self.I - self.A)

    def get_opinions(self):
        """Returns initial expressed opinions z = s (inner opinions)
        """
        return self.data_component.get_opinions()

    def get_opinion_mean(self):
        return self.data_component.get_opinion_mean()

    def propagate_message_at_equilibrium(self, message, node_id):
        """Returns expressed opinions z at equilibrium by
           setting inner opinion (s[id]) of node_id to message.
        """
        self.s[node_id, :] = message
        z_eq = self.equilibrium_mtx @ self.s
        opinion_shift_vec = z_eq - self.z
        self.z = z_eq # Update current opinion to the equilibrium opinions
        self.data_component.opinions = np.array(z_eq).flatten() # Update data object
        return np.array(opinion_shift_vec).flatten()
    
    def propagate_message_one_step(self, message, node_id):
        """Returns expressed opinions z at one step ahead by
           setting current expressed opinion (z[id]) of node_id to message.
        """
        self.z[node_id, :] = message
        z_next =  1/2 * (self.A @ self.z + self.s) # Eq. (1): (I + I)^-1 = 1/2 * 
        opinion_shift_vec = z_next - self.z
        self.z = z_next # Update current opinion to the next
        self.data_component.opinions = np.array(self.z).flatten() # Update data object
        return np.array(opinion_shift_vec).flatten()
