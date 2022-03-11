from platform import release
from keys import *
import time

def end_game():
    PressKey(ESC)
    time.sleep(10)
    ReleaseKey(ESC)
def enter_solo():
    PressKey(D)
    time.sleep(0.1)
    ReleaseKey(D)
    time.sleep(1)

    PressKey(S)
    time.sleep(1)
    ReleaseKey(S)
    time.sleep(1)

    PressKey(ENTER)
    time.sleep(0.1)
    ReleaseKey(ENTER)
    time.sleep(1)


def enter_custom():    
    PressKey(D)
    time.sleep(0.5)
    ReleaseKey(D)
    time.sleep(1)

    PressKey(S)
    time.sleep(0.5)
    ReleaseKey(S)
    time.sleep(1)
    
    PressKey(S)
    time.sleep(0.5)
    ReleaseKey(S)
    time.sleep(1)

    PressKey(S)
    time.sleep(0.5)
    ReleaseKey(S)
    time.sleep(1)
    
    PressKey(ENTER)
    time.sleep(0.5)
    ReleaseKey(ENTER)
    time.sleep(1)
    print('enter custom mode')

def start_game():
    enter_solo()
    enter_custom()
    PressKey(ENTER)
    time.sleep(0.5)
    ReleaseKey(ENTER)
    time.sleep(1)

def exit_game():
    PressKey(ESC)
    time.sleep(5)
    ReleaseKey(ESC)
    time.sleep(1)