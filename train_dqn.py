import gym
import cv2

import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gym import spaces
from floorplan_environment import FloorplanPlacementEnv
from spatial_integration import DepthmapIntegrationEvaluator, SpatialIntegrationConfig

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FloorplanGymEnv(gym.Env):
    def __init__(self):
        self.game = FloorplanPlacementEnv()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=255, shape=(50,50,4), dtype=np.uint8)
        
    def step(self, action):
        state, reward, done, info = self.game.step(action)
        return state, reward, done, info
    
    def reset(self):
        state = self.game.reset()
        return state, {}

from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODELS = True  # Save models to file so you can test later

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
INTEGRATION_DIR = os.path.join(BASE_DIR, "integration_runs")
for path in (MODEL_DIR, LOG_DIR, os.path.join(BASE_DIR, "completed"), INTEGRATION_DIR):
    os.makedirs(path, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "floorplan-dqn-")

SAVE_MODEL_INTERVAL = 500  # Save models at every X epoch
TRAIN_MODEL = True  # Set to False when evaluating a saved model.

LOAD_MODEL_FROM_FILE = False  # Load model from file
LOAD_FILE_EPISODE = 3000  # Load Xth episode from file

BATCH_SIZE = 64  # Minibatch size that select randomly from mem for train nets
MAX_EPISODE = 100000  # Max episode
MAX_STEP = 100000  # Max step size for one episode

MAX_MEMORY_LEN = 50000  # Max memory len
MIN_MEMORY_LEN = 40000  # Min memory len before start train

GAMMA = 0.97  # Discount rate
ALPHA = 0.00025  # Learning rate
EPSILON_DECAY = 0.999  # Epsilon decay rate by step

ENABLE_SPATIAL_INTEGRATION_REWARD = True
SPATIAL_INTEGRATION_REWARD_WEIGHT = 1.0
SPATIAL_INTEGRATION_EVAL_INTERVAL = 1
SPATIAL_INTEGRATION_KEEP_ARTIFACTS = False
DEPTHMAPX_CLI = os.environ.get(
    "DEPTHMAPX_CLI",
    r"C:\Users\sucre\Desktop\depthmapX\depthmapXcli_win64.exe",
)
QGIS_PROCESS = os.environ.get(
    "QGIS_PROCESS",
    r"C:\OSGeo4W\apps\qgis\bin\qgis_process.exe",
)
QGIS_ENV_BAT = os.environ.get("QGIS_ENV_BAT", r"C:\OSGeo4W\bin\o4w_env.bat")

class DuelingCNN(nn.Module):
    """
    Convolutional Q-network with dueling value and advantage heads.
    """
    def __init__(self, h, w, output_size):
        super(DuelingCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=1, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=1, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)
        linear_input_size = convw * convh * 64  # Last conv layer's out sizes
        # linear_input_size = 256
        # print('linear input size: ', linear_input_size)

        # Action layer
        # self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alinear1 = nn.Linear(in_features=3136, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=3136, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        """
        Calcs conv layers output image sizes
        """
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        # print('x shape check for conv1: ', x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten every batch
        # print('x shape check: ', x.shape)
        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q

class DQNAgent:
    def __init__(self, environment):
        """
        Hyperparameter and network initialization for the DQN agent.
        """
        # State size for the floorplan raster observation.
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Number of available placement actions.
        self.action_size = environment.action_space.n

        # Image pre process params
        self.target_h = 50  # Height after process
        self.target_w = 50  # Widht after process

        # self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table

        # Trust rate to our experiences
        self.gamma = GAMMA  # Discount coef for future predictions
        self.alpha = ALPHA  # Learning Rate

        # After many experinces epsilon will be 0.05
        # So we will do less Explore more Exploit
        self.epsilon = 1  # Explore or Exploit
        self.epsilon_decay = EPSILON_DECAY  # Adaptive Epsilon Decay Rate
        self.epsilon_minimum = 0.05  # Minimum for Explore

        # Deque holds replay mem.
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        # Online and target networks for Double DQN.
        self.online_model = DuelingCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model = DuelingCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        # Adam used as optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)  # To grayscale
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def act(self, state):
        """
        Get state and do action
        Two option can be selectedd if explore select random action
        if exploit ask nnet for action
        """

        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.online_model.forward(state)  # (1, action_size)
                action = torch.argmax(q_values).item()  # Returns the indices of the maximum value of all elements

        return action

    def train(self):
        """
        Train neural nets with replay memory
        returns loss and max_q val predicted from online_net
        """
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        # We get out minibatch and turn it to numpy array
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))
        # print('see state shape: ', state[0].shape)

        # Concat batches in one array
        # (np.arr, np.arr) ==> np.BIGarr
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        # Convert them to tensors
        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        # Make predictions
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        # Find selected action's q_value
        # print('state q values: ', state_q_values.shape)
        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # print('selected q value: ', selected_q_value.shape)
        # Get indice of the max value of next_states_q_values
        # Use that indice to get a q_value from next_states_target_q_values
        # We use greedy for policy So it called off-policy
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        # Use Bellman function to find expected q value
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        # Calc loss with expected_q_value and q_value
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        """
        Store every result to memory
        """
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        """
        Adaptive Epsilon means every step
        we decrease the epsilon so we do less Explore
        """
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay



environment = FloorplanGymEnv()
agent = DQNAgent(environment)
integration_evaluator = DepthmapIntegrationEvaluator(
    SpatialIntegrationConfig(
        enabled=ENABLE_SPATIAL_INTEGRATION_REWARD,
        work_dir=INTEGRATION_DIR,
        depthmapx_cli=DEPTHMAPX_CLI,
        qgis_process=QGIS_PROCESS,
        qgis_env_bat=QGIS_ENV_BAT,
        reward_weight=SPATIAL_INTEGRATION_REWARD_WEIGHT,
        keep_artifacts=SPATIAL_INTEGRATION_KEEP_ARTIFACTS,
    )
)

if LOAD_MODEL_FROM_FILE:
    agent.online_model.load_state_dict(torch.load(MODEL_PATH+str(LOAD_FILE_EPISODE)+".pkl"))

    with open(MODEL_PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
        param = json.load(outfile)
        agent.epsilon = param.get('epsilon')

    startEpisode = LOAD_FILE_EPISODE + 1

else:
    startEpisode = 1

last_100_ep_reward = deque(maxlen=100)  # Last 100 episode rewards
total_step = 1  # Cumulative sum of all steps in episodes
for episode in range(startEpisode, MAX_EPISODE):

    startTime = time.time()  # Keep time
    state, _ = environment.reset()
    state = agent.preProcess(state)  # Process image

    # Stack state . Every state contains 4 time contionusly frames
    # We stack frames like 4 channel image
    state = np.stack((state, state, state, state))

    total_max_q_val = 0  # Total max q vals
    total_reward = 0  # Total reward for each episode
    total_loss = 0  # Total loss for each episode
    for step in range(MAX_STEP):

        action = agent.act(state)
        # action = environment.action_space.sample()

        next_state, reward, done, info = environment.step(action)
        if (
            done
            and info.get("layout_complete")
            and episode % SPATIAL_INTEGRATION_EVAL_INTERVAL == 0
        ):
            integration_result = integration_evaluator.evaluate(next_state)
            reward += integration_result["reward"]
            info["spatial_integration"] = integration_result

        next_state = agent.preProcess(next_state)  # Process image

        # Stack state . Every state contains 4 time contionusly frames
        # We stack frames like 4 channel image
        next_state = np.stack((next_state, state[0], state[1], state[2]))

        # Store the transition in memory
        agent.storeResults(state, action, reward, next_state, done)  # Store to mem

        # Move to the next state
        state = next_state  # Update state

        if TRAIN_MODEL:
            # Perform one step of the optimization (on the target network)
            loss, max_q_val = agent.train()  # Train with random BATCH_SIZE state taken from mem
        else:
            loss, max_q_val = [0, 0]

        total_loss += loss
        total_max_q_val += max_q_val
        total_reward += reward
        total_step += 1
        if total_step % 1000 == 0:
            agent.adaptiveEpsilon()  # Decrase epsilon

        if done:  # Episode completed
            currentTime = time.time()  # Keep current time
            time_passed = currentTime - startTime  # Find episode duration
            current_time_format = time.strftime("%H:%M:%S", time.gmtime())  # Get current dateTime as HH:MM:SS
            epsilonDict = {'epsilon': agent.epsilon}  # Create epsilon dict to save model as file

            if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  # Save model as file
                weightsPath = MODEL_PATH + str(episode) + '.pkl'
                epsilonPath = MODEL_PATH + str(episode) + '.json'

                torch.save(agent.online_model.state_dict(), weightsPath)
                with open(epsilonPath, 'w') as outfile:
                    json.dump(epsilonDict, outfile)

            if TRAIN_MODEL:
                agent.target_model.load_state_dict(agent.online_model.state_dict())  # Update target model

            last_100_ep_reward.append(total_reward)
            avg_max_q_val = total_max_q_val / step

            outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Avg_Max_Q:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{} Info:{}".format(
                episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, time_passed, step, total_step, info
            )
            
            # outStr = "Episode:{} Time:{} Reward:{:.2f} Loss:{:.2f} Last_100_Avg_Rew:{:.3f} Epsilon:{:.2f} Duration:{:.2f} Step:{} CStep:{} Info:{}".format(
            #     episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), agent.epsilon, time_passed, step, total_step, info
            # )

            print(outStr)

            if SAVE_MODELS:
                outputPath = os.path.join(LOG_DIR, "training_log.txt")  # Save outStr to file
                with open(outputPath, 'a') as outfile:
                    outfile.write(outStr+"\n")

            break
