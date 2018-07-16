import collections
import random
import numpy as np

import menu

###
# vanilla q-learner that operates against a generic "game" object

np.set_printoptions(linewidth=300, suppress=True)


class GridEstimator(object):
    def __init__(self, frame_shape, num_actions, learning_rate=1.):
        self.learning_rate = 1.
        self.q = collections.defaultdict(lambda: np.zeros(num_actions))  # state -> estimated reward vector

    def evaluate(self, frame):
        return self.q[tuple(frame.flatten())]

    def update(self, frame, action, target):
        a = self.learning_rate
        key = tuple(frame.flatten())
        self.q[key][action] = (1. - a) * self.q[key][action] + a * target

    def print(self):
        print("  Q:")
        for k in sorted(self.q.keys()):
            print("     {} -> {}".format(k, self.q[k]))


def main():
    # set up reward matrix for a simple explicit graph
    game = menu.Menu(num_items=5, always_bottom=False)
    actions = game.actions()
    frame_shape = game.render(game.sample_scenario()).shape

    num_episodes = 1000
    discount = .9

    q = GridEstimator(frame_shape, len(actions), learning_rate=1.)

    # TODO: hold examples in episode buffer
    # TODO: maintain a separate target network

    for i in range(num_episodes):
        print("EPISODE %d" % i)
        state = game.sample_scenario()
        frame = game.render(state)
        while True:
            # pick an action
            # TODO: use current policy with probability p
            action_idx = random.randint(0, len(actions) - 1)

            # evaluate transition function
            try:
                next_state, reward = game.transition(state, actions[action_idx])
            except menu.BadTransition:
                # this action is not valid in this state -- assume no transition and no reward
                next_state = state
                reward = 0.

            # report
            print("  cur_state=%d, action=%d, next_state=%d -> reward = %d" %
                  (state.selection, actions[action_idx], -1 if next_state is None else next_state.selection, reward))

            # compute target
            target = reward
            if next_state is not None:
                next_frame = game.render(next_state)
                target += discount * np.max(q.evaluate(next_frame))

            # perform gradient update
            q.update(frame, action_idx, target)

            # if at goal state then terminate
            if next_state is None:
                break

            # move to next state
            state = next_state
            frame = next_frame

        q.print()


if __name__ == "__main__":
    main()
