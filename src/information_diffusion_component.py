from abc import ABC, abstractmethod
import numpy as np


class InformationDiffusionComponent(ABC):
    @abstractmethod
    def propagate_message(self, message, node_id):
        raise NotImplementedError()

    @abstractmethod
    def receive_message(self, message, node_id):
        raise NotImplementedError()


class BoundedConfidenceDiffusionComponent(InformationDiffusionComponent):
    def __init__(self, data_component, epsilon=0.2, mu=0.5, is_multishot=False, epsilon_backfire=np.inf):
        """epsilon: float : higher epsilon corresponds to higher user bound, hence higher possibility to update opinion in the common direction
           epsilon_backfire: float : higher epsilon_backfire corresponds to higher user backfire bound, hence lower possibility to update opinion in the opposite direction
        """
        self.data_component = data_component
        self.epsilon = epsilon
        self.mu = mu
        self.is_multishot = is_multishot
        self.epsilon_backfire = epsilon_backfire

    def get_opinions(self):
        return self.data_component.get_opinions()

    def get_opinion_mean(self):
        return self.data_component.get_opinion_mean()

    def propagate_message(self, message, node_id, susceptible_pool=None):
        if susceptible_pool is None:
            # initialize a pool of susceptible nodes
            susceptible_pool = list(range(self.data_component.get_num_nodes()))
            susceptible_pool = set(susceptible_pool)
            susceptible_pool.remove(node_id)
        # init propagation
        opinion_shift_tot = 0
        num_activated_users = 0

        activated_users = set()
        
        queue = set(self.data_component.get_neighbors(node_id)).intersection(susceptible_pool)
        while len(queue) > 0:
            neighbor_id = queue.pop()
            # exclude the node from the set of susceptible
            susceptible_pool.discard(neighbor_id)
            # send the message to the node
            activated, opinion_shift = self.receive_message(message, neighbor_id)
            # if the node activated on the message, the node will propagate the message to its neighbors
            if activated:
                activated_users.add(neighbor_id)
                # add neighbors of neighbor_id to the queue of users who received the message
                queue = queue.union(set(self.data_component.get_neighbors(neighbor_id)).intersection(susceptible_pool))
                opinion_shift_tot += opinion_shift
                num_activated_users += 1
        return opinion_shift_tot, num_activated_users, susceptible_pool, activated_users

    def receive_message(self, message, node_id):
        """Returns a tuple (activated_status: bool, opinion_shift)
        """
        init_opinion = self.data_component.get_opinion(node_id)
        disagreement = message - init_opinion
        opinion_shift = self.mu * disagreement
        if abs(disagreement) < self.epsilon:
            init_opinion += opinion_shift
            activated_status = True
            if self.is_multishot:
                self.data_component.update_opinion(node_id, init_opinion)
            return activated_status, opinion_shift
        elif abs(disagreement) > self.epsilon_backfire:
            init_opinion -= opinion_shift
            activated_status = True
            if self.is_multishot:
                self.data_component.update_opinion(node_id, init_opinion)
            return activated_status, - opinion_shift
        return False, 0
