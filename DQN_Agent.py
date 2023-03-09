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
    def __init__(self):
        self.gamma = 0.95   # discount rate
        self.epsilon = 0.1  # exploration rate
        self.learning_rate = 0.001
        self.memory = deque(maxlen=200)
        self.model = self._build_model()
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
        x = Dense(8, activation='linear')(x) #linear from 1 to 0   # soft_max 
        x = keras.activations.softmax(x)

        model = Model(inputs=[input_1, input_2], outputs=[x])
        model.compile(optimizer=optimizers.Adam(
            lr=self.learning_rate), loss='mse',  metrics=['mse']) #loss function################# |MSE-MSE'| < 0.0001 or learning rate*gredient < 0.001

        print(model.summary())

        return model

    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):  #??????? soft_max
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)

        return np.argmax(act_values[0]) # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            print(self.model.predict(next_state))
            print(np.amax(self.model.predict(next_state)[0]))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) #優化模型的函數
            #history = self.model.fit(state, target_f, epochs=10, verbose=0) 

        #return history
        #pyplot.plot(history.history['mse'])
        #pyplot.show()


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)




