import numpy as np


class Options(object):
    """
    Represents parameters that are fixed over the duration of a pong game
    """
    def __init__(self):
        self.screen_width = 64
        self.screen_height = 48
        self.bat_height = 4
        self.bat_offset = 2
        

class State(object):
    """
    Represents the state of a pong game
    """
    def __init__(self, opts):
        self.opts = opts
        self.player_pos = 0
        self.opponent_pos = 0


def render(state):
    """
    Render a game state to an image
    """
    pass


class BadTransition(Exception):
    """
    Thrown when attempting an invalid transition (e.g. "up" when bat is already at the top of screen)
    """
    pass


def transition(state, action):
    """
    Returns (new_state, reward) where new_state is None if game has ended
    """
    pass
