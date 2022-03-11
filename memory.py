import numpy as np
import os
class Memory(object):
    def __init__(self, state_dim, action_dim, capacity = 15000, load = False):
        self.capacity = capacity
        self.count = 0
        self.states=np.zeros((capacity,state_dim[0],state_dim[1],state_dim[2]),dtype = np.uint8)
        self.actions=np.zeros((capacity,action_dim),dtype = np.uint8)
        self.done = np.zeros(capacity, dtype = np.bool)
        self.startTrain = 5000
        if load:
            self.load()
        
    def push(self,state,action,done):
        idx = self.count % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.done[idx] = done
        self.count+=1
    def get_memory(self,batchSize,idx=None):
        if idx is None:
            idx = np.random.choice(min(self.capacity-1, self.count-1), batchSize)

        d = self.done[idx]
        # idx[d]-=1

        d_1 = self.done[idx-1]
        idx[d_1] -= 1

        s0 = np.concatenate([self.states[idx-1], self.states[idx]],-1)
        s1 = np.concatenate([self.states[idx], self.states[idx+1]],-1)
        a = self.actions[idx]
        return s0, a, s1, d
    def get_count(self):
        return self.count
    def save(self,path = 'save/memory/'):
        dirname = os.path.dirname(__file__)
        print('saving memory, current count is', self.count)
        np.save(os.path.join(dirname,path+'count.npy'),self.count)
        np.save(os.path.join(dirname,path+'state.npy'),self.states)
        np.save(os.path.join(dirname,path+'action.npy'),self.actions)
        np.save(os.path.join(dirname,path+'done.npy'),self.done)
    def load(self,path = 'save/memory/'):
        dirname = os.path.dirname(__file__)
        self.count = np.load(os.path.join(dirname,path+'count.npy'))
        self.states = np.load(os.path.join(dirname,path+'state.npy'))
        self.actions = np.load(os.path.join(dirname,path+'action.npy'))
        self.done = np.load(os.path.join(dirname,path+'done.npy'))
        print('loading memory, loaded memcount is', self.count)
