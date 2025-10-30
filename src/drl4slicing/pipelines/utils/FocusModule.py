import numpy as np
import random as rd

#=========================================================================================
#=========================================================================================
#=========================================================================================
#=========================================================================================

class TRFFocusModule:
    def __init__(self, active, r_parameter, degree, random_part):
        self.active = active
        self.r_parameter = r_parameter
        self.degree = degree
        self.random_part = random_part
        self.k_priority = degree - int(degree * random_part)
        self.kr_value = degree - self.k_priority


    def apply_focus(self, observation):
        if not self.active:
            return self.sequencing_operation(observation), None
        
        nodes, vnf = observation[0], observation[1]
        scores = [sum((max(node[0] - vnf[0], 0) ** self.r_parameter,
                      max(node[1] - vnf[1], 0) ** self.r_parameter,
                      max(node[2] - vnf[2], 0) ** self.r_parameter)) ** (1 / self.r_parameter) for node in nodes]
        
        indices1 = self.get_top_k_indices(scores)
        indices2 = self.select_k_random_without_duplicates(l1=[i for i in range(len(nodes))], l2=indices1)

        indices = sorted(indices1 + indices2)

        return self.sequencing_operation([[nodes[i] for i in indices], vnf]), indices


    def extract_focus(self, observation, indices):
        if not self.active:
            return self.sequencing_operation(observation)
        
        nodes, vnf = observation[0], observation[1]
        return self.sequencing_operation([[nodes[i] for i in indices], vnf])


    def get_top_k_indices(self, values_list):
        # Convert the list to a NumPy array
        arr = np.array(values_list)
        # Get the indices that would sort the array in descending order
        sorted_indices = arr.argsort()[::-1]
        # Get the top k indices
        top_k_indices = sorted_indices[:self.k_priority]
        return top_k_indices.tolist()
    

    def select_k_random_without_duplicates(self, l1, l2):
        # Remove all occurrences of l2 values from l1
        l1 = [item for item in l1 if item not in l2]
        # Check if there are enough unique values to select k random items
        assert len(l1) >= self.kr_value, "Not enough unique values to select k random items."
        # Select k random items without duplication
        selected_items = rd.sample(l1, self.kr_value)
        return selected_items
    

    def sequencing_operation(self, observation):
        seq_observation = []
        for cnode in observation[0]:
            seq_observation.append(cnode + observation[1])
        return seq_observation

#=========================================================================================
#=========================================================================================
#=========================================================================================
#=========================================================================================

if __name__=='__main__':
    pass