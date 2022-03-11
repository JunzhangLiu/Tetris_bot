from memory import *
from game_util import *
import numpy as np
import time
from model import *
from keys import *
from menu_navigate import *
keyboard = Keyboard()
screenshot_args = get_screenshot_args()
state_size = np.expand_dims(np.array(screenshot(screenshot_args)),-1).shape
print(state_size)
# mem = Memory(state_size,6,capacity = 10000)
agent = Agent(state_size,6)
end = False
# mem.load()
print('start')
time.sleep(10)
update_freq = 200
for i in range(100):
    while not end:
        time.sleep(0.05)
        img = screenshot(screenshot_args)
        img = np.expand_dims(img,-1)
        action = agent.choose_action(img)
        
        key = GAME_KEYS[action]
        keyboard.PressKey(key)
        blocks = detect_blocks(img)
        end = death(blocks)
        agent.store_memory(img,action,end)
        keyboard.ReleaseKey(key)
    
    end_game()
    agent.train()
    if not i%update_freq:
        agent.update()
    
    agent.save()

