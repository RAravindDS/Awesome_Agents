# filename: environment.py
import random

class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def get_state(self, x, y):
        return self.grid[x][y]

    def get_actions(self, x, y):
        actions = []
        if x > 0:
            actions.append("left")
        if x < self.width - 1:
            actions.append("right")
        if y > 0:
            actions.append("up")
        if y < self.height - 1:
            actions.append("down")
        return actions

    def get_next_state(self, x, y, action):
        if action == "left":
            return x - 1, y
        elif action == "right":
            return x + 1, y
        elif action == "up":
            return x, y - 1
        elif action == "down":
            return x, y + 1
        else:
            return None

    def perform_action(self, x, y, action):
        if action == "left":
            self.grid[x][y] = 0
            self.grid[x - 1][y] = 1
        elif action == "right":
            self.grid[x][y] = 0
            self.grid[x + 1][y] = 1
        elif action == "up":
            self.grid[x][y] = 0
            self.grid[x][y - 1] = 1
        elif action == "down":
            self.grid[x][y] = 0
            self.grid[x][y + 1] = 1