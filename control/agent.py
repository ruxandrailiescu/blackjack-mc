import numpy as np


class Agent():

    def __init__(self, gamma=0.99, eps=0.1):
        self.Q = {}
        self.sum_space = [i for i in range(4, 22)]
        self.dealer_show_card_space = [i+1 for i in range(10)]
        self.ace_space = [False, True]
        self.action_space = [0, 1]

        self.state_space = []
        self.returns = {}
        self.memory = []

        self.eps = eps
        self.gamma = gamma

        self.init_vals()
        self.init_policy()
    

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    state = (total, card, ace)
                    self.state_space.append(state)
                    for action in self.action_space:
                        self.Q[(state, action)] = 0
                        self.returns[(state, action)] = []


    def init_policy(self):
        policy = {}
        n = len(self.action_space)
        for state in self.state_space:
            policy[state] = [1/n for _ in range(n)]
        self.policy = policy


    def choose_action(self, state):
        action = np.random.choice(self.action_space, p=self.policy[state])
        return action


    def update_policy(self, state):
        actions = [self.Q[(state, a)] for a in self.action_space]
        a_max = np.argmax(actions)
        n_actions = len(self.action_space)
        probs = []

        for action in self.action_space:
            prob = 1 - self.eps + self.eps / n_actions if action == a_max else \
                    self.eps / n_actions
            probs.append(prob)

        self.policy[state] = probs


    def update_Q(self):
        visited = set()
        for idx, (state, action, _) in enumerate(self.memory):
            if (state, action) not in visited:
                visited.add((state, action))
                G = 0
                discount = 1
                for _, _, reward in self.memory[idx:]:
                    G += discount * reward
                    discount *= self.gamma
                self.returns[(state, action)].append(G)
        
        for (state, action) in visited:
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])
            self.update_policy(state)
        
        self.memory = []