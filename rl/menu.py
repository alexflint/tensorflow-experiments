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

    def __init__(self, selection=0, goal=0):
        self.selection = selection
        self.goal = goal

    def key(self):
        return (self.selection, self.goal)

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, rhs):
        return self.key() == rhs.key()

    def __lt__(self, rhs):
        return self.key() < rhs.key()

    def __repr__(self):
        return "State[%d, %d]" % (self.selection, self.goal)


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
        state.goal = random.randint(0, self.opts.num_items - 1)
        if self.opts.always_bottom:
            state.goal = self.opts.num_items - 1
        return state

    @property
    def actions(self):
        """
        actions returns the set of possible actions
        we use strings so that these do not get confused for action indices
        """
        return ("up", "select", "down")

    def render_easy(self, state):
        """
        Render a game state as three pixels indicating the correct action
        """
        return np.array((state.selection < state.goal, state.selection == state.goal, state.selection > state.goal), dtype=float)

    def render_points(self, state):
        """
        Render a game state as a column for the goal and a column for the current selection
        """
        frame = np.zeros((self.opts.num_items, 2))
        frame[state.selection, 0] = 1.
        frame[state.goal, 1] = 1.

    def render_rects(self, state):
        """
        Render a game state as a series of rectangles for each menu item
        """
        frame = np.zeros((self.opts.item_width, self.opts.item_height * self.opts.num_items + 1), dtype=int)
        frame[:, state.selection * self.opts.item_height: (state.selection + 1) * self.opts.item_height] += 1
        frame[:, state.goal * self.opts.item_height: (state.goal + 1) * self.opts.item_height] -= 1
        frame[:, -1] = state.selection - state.goal
        return frame

    def render(self, state):
        """Render a game state to an image"""
        return self.render_easy(state)

    def transition(self, state, action):
        """
        Returns (new_state, reward) where new_state is None if game has ended
        """
        if action == "select" and state.selection == state.goal:
            return None, 1.
        elif action == "select" or state.selection == state.goal:
            raise BadTransition()
        else:
            new_state = State()
            dx = -1 if action == "up" else 1
            new_state.selection = max(min(state.selection + dx, self.opts.num_items-1), 0)
            new_state.goal = state.goal
            return new_state, 0.

    def statespace(self):
        """
        Returns a list of all possible game states
        """
        out = []
        for selection in range(self.opts.num_items):
            for target in range(self.opts.num_items):
                out.append(State(selection, target))
        return out

    def format_trajectory(self, trajectory):
        """
        Produce a nice string representation of a sequence of (state, action) pairs
        """
        if len(trajectory) == 0:
            return "<empty trajectory>"
        seq = " -> ".join(str(state.selection) for state, _ in trajectory)
        return "[Goal={}] {}".format(trajectory[0][0].goal, seq)
