from .abstract import Agent


class Sarsa(Agent):
    def __init__(self, q, policy, gamma=1):
        self.q = q
        self.policy = policy
        self.gamma = gamma
        self.env = None
        self.previous_state = None
        self.previous_action = None

    def act(self, state, reward):
        action = self.policy(state)
        if self.previous_state:
            td_error = (
                reward
                + self.gamma * self.q.eval(state, action)
                - self.q(self.previous_state, self.previous_action)
            )
            self.q.reinforce(td_error)
        self.previous_state = state
        self.previous_action = action
        return action
