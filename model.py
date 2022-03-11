import tensorflow as tf 
from tensorflow import keras
from memory import *
import numpy as np
from game_util import *

def compute_reward(s0,a,s1):
    state_0 = np.expand_dims(s0[:,:,1],-1)
    state_1 = np.expand_dims(s1[:,:,0],-1)

    rewards = []
    for i in range(s0.size[0]):
        r=0
        if a[i] == 4:
            r += 0.1
        blocks_0 = detect_blocks(state_0[i])
        blocks_1 = detect_blocks(state_1[i])
        height_0,height_1 = 0,0
        cavity_0,cavity_1 = 0,0
        for j in range(blocks_0.size[0]-1):
            for k in range(blocks_0.size[1]):
                if blocks_0[j,k] == 0 and blocks_1[j+1,k] == 1:
                    cavity_0 += 1
                if height_0 != j and blocks_0[j,k] == 1:
                    height_0 = j

                if blocks_1[j,k] == 0 and blocks_1[j+1,k] == 1:
                    cavity_1 += 1
                    
                if height_1 != j and blocks_1[j,k] == 1:
                    height_1 = j

        height = (height_0 - height_1)/5
        cavity = (cavity_0 - cavity_1)/8

        if height < 0:
            height/=2

        r += (height+cavity)/2
        
        rewards.append(r)
    return np.array(rewards,dtype = np.float32)

#possible rewards: holes, elimination, passage, moving white blocks to the right position

class DuelingDeepQ(keras.Model):
    def __init__(self, actionSize):
        super(DuelingDeepQ,self).__init__()
        self.model = keras.Sequential([keras.layers.Conv2D(32, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(16, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(16, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(16, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(8, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(8, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Conv2D(8, (3,3),strides = (3,3), padding = 'valid'),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),

                                keras.layers.Flatten(),

                                keras.layers.Dense(512),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                
                                keras.layers.Dense(256),
                                keras.layers.BatchNormalization(),
                                keras.layers.ReLU(),
                                
                                ])

        self.value = keras.layers.Dense(1, kernel_initializer = keras.initializers.GlorotUniform())
        self.action = keras.layers.Dense(actionSize, kernel_initializer = keras.initializers.GlorotUniform())
    def call(self, state):
        out = self.model(state)

        value = self.value(out)
        action = self.action(out)

        Q = (value + (action - tf.math.reduce_mean(action, axis = 1, keepdims=True)))
        return Q

class Agent(object):
    def __init__(self, input_shape, action_dim,discount=0.99,start_train = 5000):
        self.memory = Memory(input_shape, action_dim)
        self.policy = DuelingDeepQ(action_dim)
        self.target = DuelingDeepQ(action_dim)
        self.action_dim = action_dim
        self.action_space = []
        for i in range(action_dim):
            self.action_space.append(i)
        self.input_shape = input_shape
        self.discount = discount
        self.eps = 0.05
        #lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(lr, 20, end_learning_rate=end_lr)
        self.opt = keras.optimizers.Adam(learning_rate=0.1)
        #self.opt = keras.optimizers.SGD(learning_rate=lr_decay,momentum=0.9)
        self.policy.compile(optimizer=self.opt,loss='mse') 
        self.start_train = start_train
        

    def initialize(self):
        _ = self.policy(tf.random.uniform((1,self.input_shape[0],self.input_shape[1],self.input_shape[2])))
        _ = self.target(tf.random.uniform((1,self.input_shape[0],self.input_shape[1],self.input_shape[2])))
        self.target.set_weights(self.policy.get_weights())

    def eval_state(self,state):
        q = self.policy(state)
        return q

    def choose_action(self, state):
        q = self.eval_state(state)
        if self.start_train > self.memory.get_count() or np.random.uniform()<self.eps:
            return np.random.choice(self.action_space)
        if np.random.uniform()<self.eps:
           return np.random.choice(self.actionSize-1)
        return np.argmax(q)
    
    def store_memory(self,st,a,done):
        self.memory.push(st,a,done)

    def train(self, batchSize=1024):
        counter = self.memory.get_count()
        if self.start_train > counter:
            print('no train, current count', counter)
            return 0
        print('start training')
        for _ in range(32):
            s0,a,s1,d = self.memory.get_memory(512)
            r = compute_reward(s0,s1,d)
            q_pred = self.policy(s0)
            q_next = tf.math.reduce_max(self.target(s1),axis=1,keepdims=True).numpy()
            q_target = np.copy(q_pred)
            for idx, terminal in enumerate(d):
                if terminal:
                    q_next[idx] = 0
                    r[idx] = 0
                q_target[idx, a[idx]] = r[idx] + self.discount * q_next[idx]
            self.policy.fit(s0,q_target)
        if self.eps >=self.epsEnd:
            self.eps-=self.eps*self.epsDecay
        return 1
        
    def save(self,path = 'save/model/'):
        self.memory.save()
        dirname = os.path.dirname(__file__)
        policy_ckpt = tf.train.Checkpoint(optimizer = self.opt, net = self.policy)
        policy_ckpt.save(os.path.join(dirname,path+'policy'))
        target_ckpt = tf.train.Checkpoint(net = self.target)
        target_ckpt.save(os.path.join(dirname,path+'target'))
        return 1


    def load(self, path = 'save/model/'):
        dirname = os.path.dirname(__file__)
        policy_ckpt = tf.train.Checkpoint(optimizer=self.opt, net = self.policy)
        policy_ckpt.restore(tf.train.latest_checkpoint(os.path.join(dirname,path+'policy-1.index')))
        target_ckpt = tf.train.Checkpoint(net = self.target)
        target_ckpt.restore(tf.train.latest_checkpoint(os.path.join(dirname,path+'target-1.index')))
        f = open(os.path.join(dirname,path+'eps.txt'), 'r')
        self.eps = float(f.readline())
        f.close()
        self.memory.load()
    def update(self):
        self.target.set_weights(self.policy.get_weights())
        print('target net updated')
    def reset_opt(self,lr):
        self.opt = keras.optimizers.Adam(lr = lr)
        self.policy.compile(optimizer=self.opt,loss='mse')
        

