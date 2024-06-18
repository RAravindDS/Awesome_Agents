# filename: agent.py
import random

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.x = random.randint(0, environment.width - 1)
        self.y = random.randint(0, environment.height - 1)

    def move(self):
        # Get the current state of the environment.
        state = self.environment.get_state(self.x, self.y)

        # Get the possible actions.
        actions = self.environment.get_actions(self.x, self.y)

        # Choose a random action.
        action = random.choice(actions)

        # The selected action must lead to a valid state
        next_state = self.environment.get_next_state(state, action)
        while next_state is None:
            action = random.choice(actions)
            next_state = self.environment.get_next_state(state, action)

        # Perform the action.
        self.environment.perform_action(state, action)

        # Update the agent's position.
        self.x, self.y = next_state