import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sb


class Board:
    def __init__(self):
        self.n_rows = 6
        self.n_cols = 25
        self.final = 40
        self.obstacles = -10
        self.sidewalk = 2
        self.litter = 5
        self.board = np.zeros([self.n_rows, self.n_cols])

    def addforth(self):
        self.board[1:self.n_rows - 2, self.n_cols - 1] = self.final
        # self.board[:, self.n_cols - 1] = self.final

    def addSideWalk(self):
        self.board[1:self.n_rows - 2, 0: self.n_cols] = self.sidewalk

    def addObstacles(self):
        p = 0.2
        n_grid = self.n_rows * self.n_cols
        indices = np.random.randint(0, n_grid, int(n_grid * p))
        for index in indices:
            x = index // self.n_cols
            y = index % self.n_cols
            if self.board[x, y] == self.litter:
                if round(np.random.uniform(0, 1), 1):
                    self.board[x, y] = self.obstacles
            elif self.board[x, y] != self.final:
                self.board[x, y] = self.obstacles

    def addLitter(self):
        p = 0.25
        n_grid = self.n_rows * self.n_cols
        indices = np.random.randint(0, n_grid, int(n_grid * p))
        for index in indices:
            x = index // self.n_cols
            y = index % self.n_cols
            if self.board[x, y] == self.obstacles:
                if round(np.random.uniform(0, 1), 1):
                    self.board[x, y] = self.litter
            elif self.board[x, y] != self.final:
                self.board[x, y] = self.litter


"""Partially adapt from cs343 Artificial Intellegence class, 
https://github.com/DD-1111/cs343-artificial-intelligence/blob/main/cs343-3-reinforcement/qlearningAgents.py"""


class QLearningAgent:
    def __init__(self, actions):
        # actions = [0, 1, 2, 3]
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.learn_epsilon = 0.85
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # <s, a, r, s'>
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # update bellman equation
        test = self.q_table
        new_q = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (new_q - current_q)

    # get action from qtable
    def get_action(self, state):
        if np.random.rand() < self.learn_epsilon:
            #  Compute the action to take in the current state.
            action = np.random.choice(self.actions)
        else:
            #  With probability self.epsilon, we should take a random action and
            #  take the best policy action otherwise.
            state_action = self.q_table[state]
            # print(state_action)
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


class GlobalState():
    def __init__(self, currentBoard):
        self.n_actions = 4
        self.board = currentBoard
        #  y,x   row,col
        self.current = np.array([np.random.randint(0, self.board.n_rows), 0])

    def reset(self):
        self.current = np.array([np.random.randint(0, self.board.n_rows), 0])

    def step(self, action):
        move = np.array([0, 0])

        if action == 0:  # up
            if self.current[0] > 0:
                move[0] -= 1
        elif action == 1:  # down
            if self.current[0] < (self.board.n_rows - 1):
                move[0] += 1
        elif action == 2:  # left
            if self.current[1] > 0:
                move[1] -= 1
        elif action == 3:  # right
            if self.current[1] < (self.board.n_cols - 1):
                move[1] += 1
        done = False

        # ***************************************************
        reward_factor = 1
        empty_space_reward = 0
        # ***************************************************
        self.current += move
        next_state = self.current

        nextval = self.board.board[next_state[0],next_state[1]]
        if nextval == self.board.litter:
            reward = reward_factor * self.board.litter
        elif nextval == self.board.obstacles:
            reward = reward_factor * self.board.obstacles
            #done = True
        elif nextval == self.board.sidewalk:
            reward = reward_factor * self.board.sidewalk
        elif nextval == self.board.final:
            reward = reward_factor * self.board.final
            done = True
        else:
            reward = reward_factor * empty_space_reward  # living cost

        return next_state, reward, done


def plotqvalue(agent, module, normbiase, title = None):
    a = agent.q_table.keys()
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    for i in range(4):
        canvas = np.zeros([6, 25])
        for key in a:
            value = agent.q_table[key][i]
            if len(key) == 5:
                y, x = int(key[1]), int(key[3:len(key) - 1])
            else:
                y, x = int(key[2]), int(key[3:len(key) - 1])
            canvas[y][x] = value
        canvas /= normbiase
        # canvas[1:4, 24] = 1

        if i == 0:
            direction = 'up'
            index = 221
        elif i == 1:
            direction = 'down'
            index = 222
        elif i == 2:
            direction = 'left'
            index = 223
        else:
            direction = 'right'
            index = 224
        ax2 = fig.add_subplot(index)
        if title == None:
            ax = sb.heatmap(canvas, vmin=0, vmax=1).set_title(
       #     f'{module} module q-value heatmap after 1000 moves, direction:{direction}')
            f'{module} module q-value heatmap after 300 episodes of training, direction:{direction}')
        else:
            ax = sb.heatmap(canvas, vmin=0, vmax=1).set_title(
                f'{title} {direction}')
    plt.show()


test = Board()
test.addSideWalk()
test.addObstacles()
test.addLitter()
test.addforth()
global_main = GlobalState(test)
agent = QLearningAgent(actions=list(range(global_main.n_actions)))
epi_x, step_y = [], []
state = str(global_main.current)
for episode in range(20):
    global_main.reset()
    state = str(global_main.current)
    step_num = 0
    while True:
        step_num += 1
        action = agent.get_action(state)
        #print(action, state)
        next_state, reward, done = global_main.step(action)
        # update qtable
        agent.learn(state, action, reward, str(next_state))
        state = str(next_state)
        # restart train once done
        if done:
            #print('--')
            break
    """for step in range(500):
        step_num += 1
        action = agent.get_action(state)
        # print(action, state)
        next_state, reward, done = global_main.step(action)
        # update qtable
        agent.learn(state, action, reward, str(next_state))
        state = str(next_state)"""


"""canvas = np.zeros([6, 25])
a = agent.q_table.keys()

for key in a:
    value = agent.q_table[key][3]
    if len(key) == 5:
        y, x = int(key[1]), int(key[3:len(key) - 1])
    else:
        y, x = int(key[2]), int(key[3:len(key) - 1])
    canvas[y][x] = value
canvas[1:4, 24] /= 6
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax1 = fig.add_subplot(221)
ax = sns.heatmap(canvas).set_title("sidewalk module's qvalue heatmap after 1000 moves, direction: right")
"""

# plot map
canvas = test.board
ax = sb.heatmap(canvas).set_title("combined module map")

plotqvalue(agent, "combined(20)", 51, title="combined module q-value heatmap after 20 episodes of training, direction:")
plt.show()


for episode in range(200):
    global_main.reset()
    state = str(global_main.current)
    step_num = 0
    while True:
        step_num += 1
        action = agent.get_action(state)
        #print(action, state)
        next_state, reward, done = global_main.step(action)
        # update qtable
        agent.learn(state, action, reward, str(next_state))
        state = str(next_state)
        # restart train once done
        if done:
            #print('--')
            break
    """for step in range(500):
        step_num += 1
        action = agent.get_action(state)
        # print(action, state)
        next_state, reward, done = global_main.step(action)
        # update qtable
        agent.learn(state, action, reward, str(next_state))
        state = str(next_state)"""
    epi_x.append(episode)
    step_y.append(step_num)

plotqvalue(agent, "combined(200)", 51, title="combined module q-value heatmap after 200 episodes of training, direction:")
plt.show()

plt.title("steps required to finish training as model get more trained\n  "
          "learning_rate = 0.1, discount_factor = 0.9, epsilon = 0.85")
plt.xlabel("training episode")
plt.ylabel("steps to finish")
plt.plot(epi_x,step_y)
plt.show()




