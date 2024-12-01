import torch
import torch.nn as nn
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import gym
import random
import copy
import torchvision.transforms as transforms
import torchvision.models as models
import imageio
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from IPython.display import Image
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



SIZE = (210, 160, 3) # Atari 2600 screen size (state space size)
WINDOW_SIZE = 3 # Number of frames to stack together as input to the network
SHOOTING_FACTOR=0.2
SHOOT_WINDOW=5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
print(device)


#Flickering POMDP will make the frame fully obscured given a probability


class ScreenObscurer:
    def __init__(self, env, probability):
        """
        Initialize the ScreenObscurer with an environment and a probability.
        
        :param env: The Gym environment to wrap.
        :param probability: Probability of obscuring the screen at each timestep.
        """
        self.env = env
        self.probability = probability

    def reset(self):
        """
        Resets the environment and returns the initial observation.
        """
        return self.env.reset()

    def step(self, action):
        """
        Executes a step in the environment and either fully obscures 
        or fully reveals the screen based on the probability.
        
        :param action: The action to take in the environment.
        :return: A tuple (observation, reward, done, info), where observation
                 might be obscured based on the probability.
        """
        obs, reward, done, info = self.env.step(action)

        # Obscure the screen based on the probability
        if np.random.rand() < self.probability:
            obs = np.zeros_like(obs)  # Fully obscure the screen
        
        return obs, reward, done, info

    def render(self):
        """
        Renders the environment.
        """
        return self.env.render()

    def close(self):
        """
        Closes the environment.
        """
        self.env.close()
        
        
        
        


#Deep memory class
class DeepMemory:
    def __init__(self, MAX_LENGTH=5000):
        # Memory is a dictionary of lists of tuples
        self.memory = {}
        self.MAX_LENGTH = MAX_LENGTH

    def remember(self, state, next_state, action, reward, done):
        # If reward is not in memory, add it
        if reward not in self.memory:
            self.memory[reward] = []

        # Add the new experience to the memory
        self.memory[reward].append((state, next_state, action, reward, done))

        # If the memory is full, remove the oldest experience
        if len(self.memory[reward]) > self.MAX_LENGTH:
            self.memory[reward].pop(0)

    def sample(self, batch_size):
        # Randomly sample from the memory
        batch = []
        for i in range(batch_size):
            reward = np.random.choice(list(self.memory.keys()))
            batch.append(random.choice(self.memory[reward]))
        return batch

    def render_sample(self, sample):
        # Render a sample from the memory
        for el in sample:
            state, next_state, action, reward, done = el
            print("--------------------------------------------------")
            plt.title("Curr State")
            plt.imshow(np.hstack(state))
            plt.axis('off')
            plt.show()

            plt.title("Next State")
            plt.imshow(np.hstack(next_state))
            plt.axis('off')
            plt.show()

            print("Action: ", action)
            print("Reward: ", reward)
            print("Done: ", done)
            print("--------------------------------------------------")
            
            
#Episodic memory class
class EpisodicMemory:
    def __init__(self, MAX_LENGTH=5000, vanilla=False):
        self.memory = []
        self.episode_memory = []
        self.MAX_LENGTH = MAX_LENGTH
        self.vanilla = vanilla

    def start_episode(self):
        self.episode_memory = []

    def end_episode(self):

        #Using vanilla reward function
        if self.vanilla:
            self.memory.append(self.episode_memory.copy())
            if len(self.memory) > self.MAX_LENGTH:
                self.memory.pop(0)

        #Using modified reward function
        else:
            # Modified reward function which incentivizes frequency of shooting, time for which the agent is alive
            last_time_shot=0
            for i in range(len(self.episode_memory)):
                if self.episode_memory[i][2] in [1]:
                    last_time_shot=i
                penalty=1/(np.exp((i-last_time_shot)/SHOOT_WINDOW)+1)
                self.episode_memory[i][3] += (len(self.episode_memory) - i)/10000 + penalty*SHOOTING_FACTOR

            self.memory.append(self.episode_memory.copy())
            if len(self.memory) > self.MAX_LENGTH:
                self.memory.pop(0)

    def remember(self, state, next_state, action, reward, done):
        self.episode_memory.append([state, next_state, action, reward, done])

    def sample(self, episode_size):
        batch = []
        if(len(self.memory) == 0):
            return batch
        for _ in range(episode_size):
            index = np.random.randint(0, len(self.memory))
            batch = batch + self.memory[index]
        return batch        
         
            
#Agent class
class Agent:
    def __init__(self, model, feature_extractor, epsMem, batchsize=64, ep_window_size=3, l_rate=0.001, epsilon_decay=0.9999):
        self.model = model # LSTM based Q Network
        self.feature_extractor = feature_extractor # ResNet18 based Transfer Learned Feature Extractor
        self.target_model = copy.deepcopy(model) # Target Network for Double DQN based learning
        self.deepMem = DeepMemory()
        self.epsMem = epsMem
        self.batchsize = batchsize
        self.ep_window_size = ep_window_size
        self.epsilon = 1.0
        self.gamma = 0.9
        
        # optimizer for both feature extractor and model
        self.optimizer = torch.optim.Adagrad(
            list(self.model.parameters()) + 
            list(filter(lambda p: p.requires_grad, self.feature_extractor.parameters())), 
            lr=l_rate)
        
        self.train_count = 0
        self.epsilon_decay = epsilon_decay


    def generate_windows(self, x, win):
        # Optimized function to generate windowed episodic memory
        y = torch.zeros(x.shape[0], win, *x.shape[1:])
        y  = torch.stack([ x[i - win + 1 : i + 1]  for i in range(WINDOW_SIZE-1, x.shape[0]) ])
        return y


    def parse_state(self, state):
        # Normalize the input frames
        normalize = transforms.Normalize(mean=[-0.445 / 0.225] * 3, std=[1 / 0.225] * 3)
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            normalize
        ])

        # Convert the input frames to tensors
        torch_images = [preprocess(img) for img in state]
        torch_images = torch.stack(torch_images)
        return torch_images

    def make_states(self, raw_states):
        states = self.parse_state(raw_states)
        states = states.to(device)
        features = self.feature_extractor(states)

        # Generate windowed episodic memory
        result = self.generate_windows(features, WINDOW_SIZE)
        return result

    def predict(self, state):
        state = self.make_states(state)
        pred = self.model(state) # predict the Q values for the current state
        # randomly take an action with epsilon probability
        if random.random() < self.epsilon:
            return random.randint(0, 6)
        else:
            # take the action predicted by the Deep Q network
            return torch.argmax(pred[0]).item()

    def train(self):
        eps_batch = self.epsMem.sample(self.ep_window_size) # sample a batch from the episodic memory
        loss_acc = 0.0
        for i in range(0, len(eps_batch), self.batchsize):
            # generate the states, next_states, actions, rewards and dones for the batch
            raw_batch = eps_batch[(max(i-WINDOW_SIZE+1,0)):(i+self.batchsize)]
            states = [x[0] for x in raw_batch]
            next_states = [x[1] for x in raw_batch]
            if i == 0:
                # if the batch is the first batch, pad the states and next_states with blank frames
                null_paddings = [np.zeros(SIZE, dtype=np.uint8) for i in range(WINDOW_SIZE-1)]
                states = null_paddings + states
                next_states = null_paddings[:-1] + [states[0]] + next_states
            else:
                # if the batch is not the first batch, pad the states and next_states with the last few frames from the previous batch
                start_batch = eps_batch[(i-WINDOW_SIZE+1):i]
                states_padding = [x[0] for x in start_batch]
                next_states_padding = [x[1] for x in start_batch]
                states = states_padding + states
                next_states = next_states_padding + next_states

            actions = [int(x[2]) for x in raw_batch]
            rewards = [float(x[3]) for x in raw_batch]
            dones = [float(int(x[4])) for x in raw_batch]

            loss_acc += self.train_batch(states, next_states, actions, rewards, dones)
        return loss_acc


    def train_batch(self, states, next_states, actions, rewards, dones):
        # Train the model using the batch
        self.feature_extractor.train()
        self.feature_extractor.zero_grad()
        states = self.make_states(states)
        next_states = self.make_states(next_states)
        future_reward = torch.max(self.target_model(next_states), 1)[0]
        dones = torch.Tensor(dones).to(device)
        rewards = torch.Tensor(rewards).to(device)
        actions = torch.Tensor(actions).long().to(device)

        # Discount the future reward by gamma
        final_reward = (rewards + future_reward * (1.0 - dones) * self.gamma).detach()
        self.model.train()
        self.model.zero_grad()
        # Predict the reward for the current state
        predicted_reward = self.model(states)
    
        actions_one_hot = torch.nn.functional.one_hot(actions, 7) #change this to 6 for bowling for assaultv-5
        # Multiply the predicted reward with the one hot encoded actions
        predicted_reward = torch.sum(predicted_reward * actions_one_hot, axis=1)
        # Calculate the loss wrt the final reward
        loss = torch.nn.functional.mse_loss(predicted_reward, final_reward)
        loss.backward() # backpropagate the loss
        self.optimizer.step() # update the weights

        self.train_count += 1

        if self.train_count % 10 == 0:
            # update the target network every 10 iterations
            self.target_model = copy.deepcopy(self.model)

        return loss.item()


#LSTM model
class LSTM(nn.Module):
    def __init__(self, inp, hidden, layers):
        super().__init__()
        self.hidden = hidden #defining no. of hidden Layers
        self.layers = layers #Defining no. of layers in an LSTM
        self.lstm = nn.LSTM(inp, hidden, layers,batch_first=True) #giving arguments as input to the LSTM
        self.fc = nn.Linear(hidden, 7) #this is 7 for the Assault-v5 game, 6 for bowling

    def forward(self, x):
        self.lstm.flatten_parameters() #Flattening the parameters
        batch_size = x.size(0)
        h0 = torch.zeros(self.layers, batch_size, self.hidden).to(device) #Initializing
        c0 = torch.zeros(self.layers, batch_size, self.hidden).to(device) #Initializing
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) #Taking the final time_step of the time sequence
        return out


#Resnet model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet= models.resnet18(pretrained=True) #Using ResNet18 pretrained model
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1]) #Removing the last layer of the ResNet18 model

        for params in self.resnet.parameters():
            params.requires_grad = False #Freezing the parameters of the ResNet18 model for transfer learning
        # self.resnet.eval()

        # Unfreeze only the last convolutional layer
        for param in list(self.resnet[-1].parameters()):  # Access the last layer of the sequential
            param.requires_grad = True

    def forward(self,x):
        return self.resnet(x).view(-1,512)
    

#Helper functions
def visualize_frames(frames):
    # Visualize the frames
    plt.imshow(np.hstack(frames))
    plt.axis('off')
    plt.show()

def visualize_rewards(rew):
    # Visualize the rewards
    ax = plt.figure(figsize=(6, 3))
    plt.plot(rew)
    plt.xlabel("Time step in Frames")
    plt.ylabel("Reward")
    plt.show()

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    # function to save the frames as a gif
    plt.figure(figsize=(frames[0].shape[1] / 36.0, frames[0].shape[0] / 36.0), dpi=36)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    # Use the 'imageio' library to save the animation as a gif
    gif_path = path + filename
    writer = imageio.get_writer(gif_path, duration=0.005)
    for i in range(len(frames)):
        writer.append_data(frames[i])
    writer.close()

    plt.cla()
    plt.clf()
    plt.close()

def initialize_model(eps_mem_size, num_lstm_hidden_layers, batch_size, ep_window_size, l_rate, epsilon_decay, vanilla = False):
  # Initialize the model, feature extractor and agent
#   epMem = DeepMemory(MAX_LENGTH=eps_mem_size)
  epMem = EpisodicMemory(MAX_LENGTH=eps_mem_size, vanilla = vanilla)
  feature_extractor = Net().to(device)
  model = LSTM(512, num_lstm_hidden_layers, 1).to(device)
  agent = Agent(model, feature_extractor, epMem, batch_size, ep_window_size, l_rate, epsilon_decay)
  return (epMem, model, agent)

def perform_train(env, agent, epMem, epsilon_decay,num_of_episodes, time_step_size, window_size, show_gifs=False, gif_show_frequency=1, csvfile_name='plot_data.csv'):
  
  
  with open(csvfile_name, 'w') as file:
      # clean-up the csv file if it already exists
      pass
  for i in range(num_of_episodes):
      # epsilon decay
      if(i<200):
          # for the first 200 episodes, epsilon decays proportional to 1/i
          agent.epsilon=1/max(1, i/10) #change this i/10 for assault-v5
      else:
          # for the rest of the episodes, epsilon decays exponentially
          agent.epsilon = 0.1*(epsilon_decay**(i-99))
          
    # Updated reset logic for compatibility
      reset_result = env.reset()
      if isinstance(reset_result, tuple):  # Newer Gym versions
          observation, info = reset_result
      else:  # Older Gym versions
          observation = reset_result
          info = {}
    
      frames = []
      obs, rew = [], []
      curr_state = [np.zeros(SIZE, dtype=np.uint8) for i in range(window_size)]
      total_reward = 0
      epMem.start_episode()
      avg_train_loss = 0
      last_life = time_step_size

      for t in tqdm(range(time_step_size)):
          frames.append(env.render(mode='rgb_array'))
          action = agent.predict(curr_state)
          observation, reward, done, info = env.step(action)
          prev_state = curr_state.copy()
          curr_state.pop(0)
          curr_state.append(observation.copy())
          obs.append(observation.copy())
          rew.append(reward)
          total_reward += reward
          epMem.remember(prev_state[-1].copy(), curr_state[-1].copy(), action, reward, done)
          if(t % 100 == 0):
            avg_train_loss += agent.train()
          if done:
              last_life = t
              break

      epMem.end_episode()
      print("Avg Train Loss:", avg_train_loss/time_step_size)
      print("Total Reward:", total_reward)
      print("Exploration Factor: epsilon:",agent.epsilon)
      print("--------------------------------------------------")
      with open(csvfile_name, 'a') as file:
            x = [total_reward, avg_train_loss/time_step_size, agent.epsilon, last_life]
            x = [str(i) for i in x]
            file.write(','.join(x) + '\n')

      if(show_gifs and i % gif_show_frequency == 0):
          save_frames_as_gif(frames, filename=f'gym_animation_{i//gif_show_frequency}.gif')
          display(Image(data=open(f'gym_animation_{i//gif_show_frequency}.gif','rb').read(), format='png'))
                  
def train_stepLR(env, agent, epMem, epsilon_decay, num_of_episodes, time_step_size, window_size, show_gifs=False, gif_show_frequency=1, csvfile_name='plot_data.csv'):
  
  
  scheduler = StepLR(agent.optimizer, step_size=25, gamma=0.8)
  with open(csvfile_name, 'w') as file:
      pass
  for i in range(num_of_episodes):
      if(i<200):
            agent.epsilon=1/max(1, i/10)
      else:
           agent.epsilon = 0.05*(epsilon_decay**(i-199))

      # Updated reset logic for compatibility
      reset_result = env.reset()
      if isinstance(reset_result, tuple):  # Newer Gym versions
          observation, info = reset_result
      else:  # Older Gym versions
          observation = reset_result
          info = {}

      # observation, info = env.reset()
      frames = []
      obs, rew = [], []
      curr_state = [np.zeros(SIZE, dtype=np.uint8) for i in range(window_size)]
      total_reward = 0
      epMem.start_episode()
      avg_train_loss = 0
      last_life = time_step_size

      for t in range(time_step_size):
          frames.append(env.render(mode='rgb_array'))
          action = agent.predict(curr_state)
          observation, reward, done, info = env.step(action)
          prev_state = curr_state.copy()
          curr_state.pop(0)
          curr_state.append(observation.copy())
          obs.append(observation.copy())
          rew.append(reward)
          total_reward += reward
          # done = terminated or truncated
          # deepMem.remember(prev_state, curr_state.copy(), action, reward, done)
          epMem.remember(prev_state[-1].copy(), curr_state[-1].copy(), action, reward, done)
          if(t % 100 == 0):
            avg_train_loss += agent.train()
          if done:
              last_life = t
              break

      scheduler.step()
      epMem.end_episode()
      print(f'Episode: {i+1}, Number of steps: {last_life}, Total Reward: {total_reward}')
      with open(csvfile_name, 'a') as file:
            # log the data to the csv file for plotting subsequently
            x = [total_reward, avg_train_loss/time_step_size, agent.epsilon, last_life, scheduler.get_lr()]
            x = [str(i) for i in x]
            file.write(','.join(x) + '\n')

      if(show_gifs and i % gif_show_frequency == 0):
          # save the frames as a gif if show_gifs is True and at a frequency of gif_show_frequency
          save_frames_as_gif(frames, filename=f'gym_animation_{i//gif_show_frequency}.gif')
          display(Image(data=open(f'gym_animation_{i//gif_show_frequency}.gif','rb').read(), format='png'))       
          
def train_cosine(env, agent, epMem, epsilon_decay, num_of_episodes, time_step_size, window_size, show_gifs=False, gif_show_frequency=1, csvfile_name='plot_data.csv'):
  scheduler = CosineAnnealingWarmRestarts(agent.optimizer, T_0=10, T_mult=1, eta_min=0.001)
  with open(csvfile_name, 'w') as file:
      pass
  for i in range(num_of_episodes):
      if(i<200):
            agent.epsilon=1/max(1, i/10)
      else:
           agent.epsilon = 0.05*(epsilon_decay**(i-199))

      # Updated reset logic for compatibility
      reset_result = env.reset()
      if isinstance(reset_result, tuple):  # Newer Gym versions
          observation, info = reset_result
      else:  # Older Gym versions
          observation = reset_result
          info = {}
          
      frames = []
      obs, rew = [], []
      curr_state = [np.zeros(SIZE, dtype=np.uint8) for i in range(window_size)]
      total_reward = 0
      epMem.start_episode()
      avg_train_loss = 0
      last_life = time_step_size

      for t in range(time_step_size):
          frames.append(env.render(mode='rgb_array'))
          action = agent.predict(curr_state)
          observation, reward, done, info = env.step(action)
          prev_state = curr_state.copy()
          curr_state.pop(0)
          curr_state.append(observation.copy())
          obs.append(observation.copy())
          rew.append(reward)
          total_reward += reward
        #   deepMem.remember(prev_state, curr_state.copy(), action, reward, done)
          epMem.remember(prev_state[-1].copy(), curr_state[-1].copy(), action, reward, done)
          if(t % 100 == 0):
            avg_train_loss += agent.train()
          if done:
              last_life = t
              break
      scheduler.step()


      epMem.end_episode()
      print(f'Episode: {i+1}, Number of steps: {last_life}, Total Reward: {total_reward:.2f}')
      with open(csvfile_name, 'a') as file:
            x = [total_reward, avg_train_loss/time_step_size, agent.epsilon, last_life, scheduler.get_lr()]
            x = [str(i) for i in x]
            file.write(','.join(x) + '\n')

      if(show_gifs and i % gif_show_frequency == 0):
          save_frames_as_gif(frames, filename=f'gym_animation_{i//gif_show_frequency}.gif')
          display(Image(data=open(f'gym_animation_{i//gif_show_frequency}.gif','rb').read(), format='png'))
        
        
        
        
