from __future__ import absolute_import
from __future__ import print_function 

from tensorflow import optimizers  # prevent from random forcasting, improve efficiency

import keras # important tool for deep learning
import random
import numpy as np

from collections import deque  #容器
from keras.layers import Input, Flatten, Dense
from keras.models import Model  


class DQNAgent:
    memory = deque(maxlen=500)

    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.action_size = 8

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        input_1 = Input(shape=(24, 1))
        x1 = Flatten()(input_1)

        input_2 = Input(shape=(8, 1))
        x2 = Flatten()(input_2)

        x = keras.layers.concatenate([x1, x2])
        x = Dense(128, activation='relu')(x) #Dense 普通神經網路  ## why 128
        x = Dense(64, activation='relu')(x)  # why 64
        x = Dense(8)(x) #linear from 1 to 0   # soft_max 

        model = Model(inputs=[input_1, input_2], outputs=[x])
        model.compile(optimizer=optimizers.Adam(
            lr=self.learning_rate), loss='mse',  metrics=['mse']) #loss function################# |MSE-MSE'| < 0.0001 or learning rate*gredient < 0.001

        #print(model.get_weights())
        
        return model

    def remember(self, state, action, reward, next_state, done): 
        DQNAgent.memory.append((state, action, reward, next_state, done))

    def act(self, state):  #??????? soft_max
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(DQNAgent.memory, batch_size)
        A = 0
        for state, action, reward, next_state, done in minibatch:
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = (reward + self.gamma *
                          np.amax(self.target_model.predict(next_state)[0]))
            A += (target[0][action] - Q_future)**2
            target[0][action] = Q_future
            self.model.fit(state, target, epochs=1, verbose=0) #優化模型的函數
            
        MSE = A/batch_size
            
        return MSE

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
            


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




