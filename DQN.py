#%%
from os import replace
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.layers.core import Activation, Dropout
from collections import deque
import customTensorBoard
import time
import numpy as np
import random

MIN_REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "DQN_snake"
REPLAY_MEMORY_SIZE = 50000
MINI_BATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5


class DQNAgent:

    def create_model(self,input_shape):

        model = Sequential()
        
        model.add(Conv2D(128,(3,3),input_shape = input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256,(3,3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(2,2))

        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(4,activation='linear'))

        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

        return model

    def __init__(self,input_shape = (11,21,1)):

        self.model = self.create_model(input_shape)
        
        self.target_model = self.create_model(input_shape)
        self.target_model.set_weights(self.model.get_weights())
        #Is this line really useful for initialisation ?

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = customTensorBoard.ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        print(f"erreur get_qs LALALALALALALALAAL  shape state : {np.array(state).shape}")
        return self.model.predict(np.array(state).reshape(-1, *np.array(state).shape,1))


    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 

        minibatch = random.sample(self.replay_memory,MINI_BATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch]).reshape(64,11,21,1)
        #print(f"shape avant erreur 1 de current_states : {current_states.shape}")
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]).reshape(64,11,21,1)
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
        self.model.fit(np.array(X).reshape(64,11,21,1), np.array(y), batch_size=MINI_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
#%%


# %%
