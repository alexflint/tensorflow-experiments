# menu is a game in which the player is rewarded for selecting a highlighted menu item

import numpy as np
import random


class Options(object):
    """
    Represents parameters that are fixed over the duration of a pong game
    """

    def __init__(self):
        self.num_items = 5
        self.item_width = 1
        self.item_height = 2
        self.always_bottom = False


class State(object):
    """
    Represents the state of a pong game
    """

    def __init__(self, selection=0, target=0):
        self.selection = selection
        self.target = target

    def key(self):
        return (self.selection, self.target)

    def __hash__(self):
        return hash(self.key())
    
    def __eq__(self, rhs):
        return self.key() == rhs.key()

    def __lt__(self, rhs):
        return self.key() < rhs.key()

    def __repr__(self):
        return "State[%d, %d]" % (self.selection, self.target)


class BadTransition(Exception):
    """
    Thrown when attempting an invalid transition (e.g. "up" when bat is already at the top of screen)
    """
    pass


class Menu(object):
    def __init__(self, num_items=5, item_width=1, item_height=2, always_bottom=True):
        self.opts = Options()
        self.opts.num_items = num_items
        self.opts.item_width = item_width
        self.opts.item_height = item_height
        self.opts.always_bottom = always_bottom

    def sample_scenario(self):
        """
        Sample a new "level" and return an initial state
        """
        state = State()
        state.selection = random.randint(0, self.opts.num_items - 1)
        state.target = random.randint(0, self.opts.num_items - 1)
        if self.opts.always_bottom:
            state.target = self.opts.num_items - 1
        return state

    def actions(self):
        """
        actions returns the set of possible actions
        """
        return (-1, 0, 1)  # 0 mean "select"

    def render(self, state):
        """
        Render a game state to an image
        """
        frame = np.zeros((self.opts.item_width, self.opts.item_height * self.opts.num_items), dtype=int)
        frame[:, state.selection * self.opts.item_height: (state.selection + 1) * self.opts.item_height] += 1
        frame[:, state.target * self.opts.item_height: (state.target + 1) * self.opts.item_height] -= 1
        return frame

    def transition(self, state, action):
        """
        Returns (new_state, reward) where new_state is None if game has ended
        """
        if action == 0 and state.selection == state.target:
            return None, 1.
        elif action == 0 or state.selection == state.target:
            raise BadTransition()
        else:
            new_state = State()
            new_state.selection = max(min(state.selection + action, self.opts.num_items-1), 0)
            new_state.target = state.target
            return new_state, 0.
