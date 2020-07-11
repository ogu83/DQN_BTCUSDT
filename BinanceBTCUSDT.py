import math, random, os, multiprocessing, time

import pandas as pd
import numpy as np

from tqdm import tqdm as tqdm
from joblib import Parallel, delayed
from collections import deque
from datetime import date

# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
from matplotlib.pyplot import figure

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
# from keras.callbacks import TensorBoard
import tensorflow as tf

import FinancialIndicators as fi, BinanceHelper as bh

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 4  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 1  # How many steps (samples) to use for training

MIN_REWARD = 0  # For model save
MEMORY_FRACTION = 0.20
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

# Environment settings
EPISODES = 200

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = True

#Construct Binance Client
from binance.client import Client
api_key = ''
api_secret = ''
client = Client(api_key, api_secret)

#Get Data
symbol = "BTCUSDT"
start = "1 Jan, 2017"
interval = Client.KLINE_INTERVAL_1WEEK

MODEL_NAME = symbol + "_" + interval
DF_FILENAME = "./Data/DF_" + symbol + "_" + str(date.today()) + "_" + interval

print("Get Binance Data", interval)

if os.path.exists(DF_FILENAME):
    print("Read df from file")
    df = pd.read_pickle(DF_FILENAME)    
else:
    klines = np.array(bh.get_historical_klines(client, symbol, interval, start))

    df = bh.binanceDataFrame(klines)
    print(df.shape)

    # Preprocessing
    USE_CPU_PERCENTAGE = 100
    num_cores = int(multiprocessing.cpu_count() * (USE_CPU_PERCENTAGE / 100))
    if num_cores < 1:
        num_cores=1

    # print("Preprocessing ... Num of Cores: ", num_cores)

    remove_cols = [c for c in df.columns if c not in ['Open', 'Close', 'High', 'Low', 'Volume']]
    df.drop(remove_cols, axis=1, inplace=True)

    windows = [3, 5, 9, 14, 21]

    df = fi.stochastic_oscillator_k(df)
    df = fi.ppsr(df)  
    df = fi.mass_index(df)
    df = fi.chaikin_oscillator(df)
    df = fi.ultimate_oscillator(df)

    def processIndicatorsWindow(w, df):
        df = fi.moving_average(df,w)
        df = fi.exponential_moving_average(df,w)
        df = fi.momentum(df,w)
        df = fi.rate_of_change(df,w)
        df = fi.average_true_range(df,w)
        df = fi.bollinger_bands(df,w)
        df = fi.stochastic_oscillator_d(df,w)
        df = fi.trix(df,w)
        df = fi.average_directional_movement_index(df,w,w*2)
        df = fi.macd(df,w,w*2)  
        df = fi.vortex_indicator(df,w)
        df = fi.kst_oscillator(df,w+1,w+2,w+3,w+4,w,w*2,w*3,w*4)
        df = fi.relative_strength_index(df,w)
        df = fi.true_strength_index(df,w,w*2)
        df = fi.accumulation_distribution(df,w)
        df = fi.money_flow_index(df,w)
        df = fi.on_balance_volume(df,w)
        df = fi.force_index(df,w)
        df = fi.ease_of_movement(df,w)
        df = fi.commodity_channel_index(df,w)
        df = fi.coppock_curve(df,w)
        df = fi.keltner_channel(df,w)
        df = fi.donchian_channel(df,w)
        df = fi.standard_deviation(df,w)  

    print("Process Indicators at Windows", windows)

    # Parallel(n_jobs=num_cores)(delayed(processIndicatorsWindow)(w,df) for w in tqdm(windows))  

    for w in tqdm(windows):
        df = fi.moving_average(df,w)
        df = fi.exponential_moving_average(df,w)
        df = fi.momentum(df,w)
        df = fi.rate_of_change(df,w)
        df = fi.average_true_range(df,w)
        df = fi.bollinger_bands(df,w)  
        df = fi.stochastic_oscillator_d(df,w)
        df = fi.trix(df,w)
        df = fi.average_directional_movement_index(df,w,w*2)
        df = fi.macd(df,w,w*2)    
        df = fi.vortex_indicator(df,w)
        df = fi.kst_oscillator(df,w+1,w+2,w+3,w+4,w,w*2,w*3,w*4)
        df = fi.relative_strength_index(df,w)
        df = fi.true_strength_index(df,w,w*2)
        df = fi.accumulation_distribution(df,w)
        df = fi.money_flow_index(df,w)
        df = fi.on_balance_volume(df,w)
        df = fi.force_index(df,w)
        df = fi.ease_of_movement(df,w)
        df = fi.commodity_channel_index(df,w)
        df = fi.coppock_curve(df,w)
        df = fi.keltner_channel(df,w)
        df = fi.donchian_channel(df,w)
        df = fi.standard_deviation(df,w)  

    print("Process Indicators completed")

    indicator_cols = [c for c in df.columns if c not in ['Open', 'Close', 'High', 'Low']]
    for indicator in indicator_cols:
        df[indicator] = df[indicator].shift(1)

    df = df[max(windows)+1:]
    df.fillna(value=0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    pd.to_pickle(df, DF_FILENAME)
    print("Write df to file")

print(df.shape)
print(df.describe())

##### Q LEARNING #####

class BTCUSDT_Env:

    ACTION_SPACE_SIZE = 3

    def __init__(self, df, BTC_Balance=1, USDT_Balance=5_000, ratio = 0.1, min_usd = 10, min_btc = 1/1_000):
        self.ratio = ratio

        self.min_usd = min_usd
        self.min_btc = min_btc

        self.df = df

        self.Init_BTC_Balance = BTC_Balance
        self.Init_USDT_Balance = USDT_Balance

        self.reset()        

    def __str__(self):
        return f"Balance: {self.BTC_Balance:.5f}BTC, {self.USDT_Balance:.0f}USD | Total Assets: {self.Total_Asset_BTC:.5f}BTC | Shape: {df.shape}"

    def observation_cols(self):
        return [c for c in self.df.columns if c not in ['Close', 'High', 'Low']]

    def reset(self):
        self.BTC_Balance = self.Init_BTC_Balance
        self.USDT_Balance = self.Init_USDT_Balance
        
        self.Init_Total_Asset_BTC = self.Init_BTC_Balance + self.Init_USDT_Balance / df.iloc[0]['Open']
        self.Total_Asset_BTC = self.Init_Total_Asset_BTC

        self.BTC_Balance_History = []
        self.USDT_Balance_History = []
        self.Profit_BTC_History = []
        self.Total_Asset_BTC_History = []

        self.Total_Asset_BTC_History.append(self.Total_Asset_BTC)
        self.BTC_Balance_History.append(self.BTC_Balance)
        self.USDT_Balance_History.append(self.USDT_Balance)
        self.Profit_BTC_History.append(0)

        self.episode_step = 0
                
        observation = np.array(df[self.observation_cols()].iloc[self.episode_step])
        return observation

    def step(self, action):
        self.episode_step += 1

        # actions: -1=Sell (Buy USDT), 0=DoNothing, 1=Buy (Buy BTC)
        if (action == -1 and self.BTC_Balance > self.min_btc):
            amount_BTC =  self.BTC_Balance * self.ratio
            amount_USDT = amount_BTC * df.iloc[self.episode_step]['Open']
            self.BTC_Balance -= amount_BTC
            self.USDT_Balance += amount_USDT
        elif (action == 1 and self.USDT_Balance > self.min_usd):
            amount_USDT = self.USDT_Balance * self.ratio
            amount_BTC = amount_USDT / df.iloc[self.episode_step]['Open']
            self.BTC_Balance += amount_BTC
            self.USDT_Balance -= amount_USDT            
            
        self.Total_Asset_BTC = self.BTC_Balance + self.USDT_Balance / df.iloc[self.episode_step]['Open']

        change = 0
        if (action == 1):
            change = 100 * (df.iloc[self.episode_step]['Close'] - df.iloc[self.episode_step]['Open']) / df.iloc[self.episode_step]['Open']
        elif (action == -1):
            change = 100 * (df.iloc[self.episode_step]['Open'] - df.iloc[self.episode_step]['Close']) / df.iloc[self.episode_step]['Open']
        
        profit_btc = 100 * (self.Total_Asset_BTC - self.Init_Total_Asset_BTC) / self.Init_Total_Asset_BTC        

        reward = profit_btc + self.ratio * change

        done = False
        if self.Total_Asset_BTC < self.min_btc or self.episode_step == (len(self.df)-1):
            done = True

        self.Total_Asset_BTC_History.append(self.Total_Asset_BTC)
        self.BTC_Balance_History.append(self.BTC_Balance)
        self.USDT_Balance_History.append(self.USDT_Balance)
        self.Profit_BTC_History.append(profit_btc)

        new_observation = np.array(df[self.observation_cols()].iloc[self.episode_step])

        return new_observation, reward, done          

    def render(self):
        figure(num=None, figsize=(18, 30), dpi=80, facecolor='silver', edgecolor='gray')

        plt.subplot(5, 1, 1)
        plt.plot(self.df['Open'])
        plt.xlabel('time')
        plt.ylabel('($)')
        plt.title('Open Price Graph')
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(self.BTC_Balance_History)
        plt.xlabel('time')
        plt.ylabel('BTC')
        plt.title('BTC Balance')
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(self.USDT_Balance_History)
        plt.xlabel('time')
        plt.ylabel('USD')
        plt.title('USD Balance')
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(self.Total_Asset_BTC_History)
        plt.xlabel('time')
        plt.ylabel('BTC')
        plt.title('Total BTC Balance')
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(self.Profit_BTC_History)
        plt.xlabel('time')
        plt.ylabel('%')
        plt.title('Profit in BTC')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

env = BTCUSDT_Env(df)
print(env)

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# Agent class
class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        # self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)        

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (3)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X),
                       np.array(y), 
                       batch_size=MINIBATCH_SIZE, 
                       verbose=0, 
                       shuffle=False
                       #callbacks=[self.tensorboard] if terminal_state else None
                       )

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state)) - 1 
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE) - 1

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY and done:
            env.render()
        
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)
        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
