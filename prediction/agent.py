import numpy as np


# Learning the state-value function for a given policy
class Agent():

    def __init__(self, gamma=1.0):
        self.V = {}
        self.sum_space = [i for i in range(4, 22)]
        self.dealer_show_card_space = [i+1 for i in range(10)]
        self.ace_space = [False, True]  # usable ace
        self.action_space = [0, 1]  # stick or hit

        self.state_space = []   # 3-tuple with player sum, dealer card, and usable ace flag
        self.returns = {}   # list of returns for each state
        self.memory = []    # states and rewards
        self.gamma = gamma

        self.init_vals()
    
    def init_vals(self):
        for total in self.sum_space:
            for dealer_card in self.dealer_show_card_space:
                for usable_ace in self.ace_space:
                    self.V[(total, dealer_card, usable_ace)] = 0
                    self.returns[(total, dealer_card, usable_ace)] = []
                    self.state_space.append((total, dealer_card, usable_ace))

    def policy(self, state):
        total, _, _ = state
        action = 0 if total >= 20 else 1
        return action
    
    def update_V(self):     # when an episode ends
        visited = set()
        for idx, (state, _) in enumerate(self.memory):
            if state not in visited:
                visited.add(state)
                G = 0
                discount = 1
                for _, reward in self.memory[idx:]:
                    G += discount * reward
                    discount *= self.gamma
                self.returns[state].append(G)
        
        for state in visited:
            self.V[state] = np.mean(self.returns[state])

        self.memory = []