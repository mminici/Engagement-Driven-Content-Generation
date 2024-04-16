from abc import ABC, abstractmethod
import networkx as nx
from sklearn.preprocessing import normalize, StandardScaler
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
        
    
    def get_opinions(self):
        """Returns initial expressed opinions z = s (inner opinions)
        """
        return self.data_component.get_opinions()

    def get_opinion_mean(self):
        return self.data_component.get_opinion_mean()

    def precompute_equilirbium_mtx(self, node_id):
        """Removes in connections of node_id row
        """
        self.A[node_id, :] = 0
        self.equilibrium_mtx = np.linalg.inv(2 * self.I - self.A)

    def propagate_message_at_equilibrium(self, message, node_id, update_opinions = False):
        """Returns expressed opinions z at equilibrium by
           setting inner opinion (s[id]) of node_id to message.
        """
        self.s[node_id, :] = message
        scaled_s = StandardScaler().fit_transform(self.s)
        z_eq = self.equilibrium_mtx @ scaled_s
        return z_eq
        
        # scaled_z = StandardScaler().fit_transform(self.z)
        # opinion_shift_vec = z_eq - scaled_z
        # if update_opinions:
        #     self.z = z_eq # Update current opinion to the equilibrium opinions
        # self.data_component.opinions = np.array(z_eq).flatten() # Update data object
        # return np.array(opinion_shift_vec).flatten()
    
    def propagate_message_one_step(self, message, node_id):
        pass
    
    def polarization_plus_disagreement_at_equilibrium(self, message, node_id):
        """Returns the sum of polarization and disagreement at equilibrium given the influence of the 
            immutable opinion of the LLM (node_id)
        """
        # 1. assign message to llm
        self.s[node_id, :] = message
        scaled_s = StandardScaler().fit_transform(self.s)

        # 2. LLM does not update opinion -> remove LLM's infuencers
        self.A[node_id, :] = 0

        # 3. Compute in one-shot the polarization plus disagreement at equilibirum
        D_in = np.diag(np.asarray(self.A.sum(axis=0)).flatten())
        I = np.identity(D_in.shape[0])

        z_1 = np.linalg.inv(2*I-self.A.T) @ scaled_s
        z_2 = np.linalg.inv(2*I-self.A) @ scaled_s
        z_3 = np.linalg.inv(2*I-self.A.T) @ (D_in - I) @ z_2

        return (scaled_s.T @ z_1 + 1/2 * scaled_s.T @ z_3).item() # 1/2 correction to the wrong formula (14) in Cinus et al

    @staticmethod
    def polarization(z) -> float:
        return np.sum(np.asarray(z).flatten()**2)
    
    @staticmethod
    def disagreement(A, z) -> float:
        D_in = np.diag(np.asarray(A.sum(axis=0)).flatten())
        return (1 / 2 * z.T @ (D_in + np.identity(D_in.shape[0]) - 2 * A) @ z).item()