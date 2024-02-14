from abc import ABC, abstractmethod


class InformationDiffusionComponent(ABC):
    @abstractmethod
    def propagate_message(self, message, node_id):
        raise NotImplementedError()

    @abstractmethod
    def receive_message(self, message, node_id):
        raise NotImplementedError()


class BoundedConfidenceDiffusionComponent(InformationDiffusionComponent):
    def __init__(self, data_component, epsilon=0.2, mu=0.5):
        self.data_component = data_component
        self.epsilon = epsilon
        self.mu = mu

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
        queue = set(self.data_component.get_neighbors(node_id)).intersection(susceptible_pool)
        while len(queue) > 0:
            neighbor_id = queue.pop()
            # exclude the node from the set of susceptible
            susceptible_pool.discard(neighbor_id)
            # send the message to the node
            activated, opinion_shift = self.receive_message(message, neighbor_id)
            # if the node activated on the message, the node will propagate the message to its neighbors
            if activated:
                # add neighbors of neighbor_id to the queue of users who received the message
                queue = queue.union(set(self.data_component.get_neighbors(neighbor_id)).intersection(susceptible_pool))
                opinion_shift_tot += opinion_shift
                num_activated_users += 1
        return opinion_shift_tot, num_activated_users, susceptible_pool

    def receive_message(self, message, node_id):
        init_opinion = self.data_component.get_opinion(node_id)
        disagreement = message - init_opinion
        if abs(disagreement) < self.epsilon:
            init_opinion += self.mu * disagreement
            self.data_component.update_opinion(node_id, init_opinion)
            return True, self.mu * disagreement
        return False, 0
