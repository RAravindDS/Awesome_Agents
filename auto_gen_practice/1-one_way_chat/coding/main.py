# filename: main.py
from agent import Agent
from environment import Environment

# Create an environment.
environment = Environment(5, 5)

# Create an agent.
agent = Agent(environment)

# Run the agent.
for i in range(100):
    agent.move()