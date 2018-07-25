import collections
import random
import numpy as np

import menu

###
# vanilla q-learner that operates against a generic "game" object

np.set_printoptions(linewidth=300, suppress=True)


class GridEstimator(object):
    def __init__(self, frame_shape, num_actions, learning_rate=1.):
        self.learning_rate = learning_rate
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


class LinearEstimator(object):
    def __init__(self, frame_shape, num_actions, learning_rate=1.):
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.feature_size = len(self.feature(np.zeros(frame_shape)))
        self.q = np.zeros((num_actions, self.feature_size))

    def feature(self, frame):
        # prepend 1 so that the weight vector includes a bias term
        return np.concatenate(([1], frame.flatten()))

    def evaluate(self, frame):
        return np.dot(self.q, self.feature(frame))

    def update(self, frame, action, target):
        a = self.learning_rate
        x = self.feature(frame)
        r = (np.dot(self.q[action], x) - target)
        j = x
        self.q[action] -= a * j * r
        # print("  update: ========")
        # print("    action: ", action)
        # print("    feature: ", x)
        # print("    target: ", target)
        # print("    prediction: ", np.dot(before, x))
        # print("    residual: ", r)
        # print("    jacobian:", j)
        # print("    gradient: ", j*r)
        # print("    q[action]: ", before, "->", self.q[action])

    def solve(self, frames, actions, targets):
        assert(len(frames) == len(actions) == len(targets))
        m = np.zeros((len(frames), self.num_actions * self.feature_size))
        b = np.zeros(len(frames))
        for i, (f, a, t) in enumerate(zip(frames, actions, targets)):
            m[i, a*self.feature_size: (a+1) * self.feature_size] = self.feature(f)
            b[i] = t
        print("solving:")
        for f, a, t in zip(frames, actions, targets):
            print("  %s -> %s -> %s" % (str(f), str(a), str(t)))
        # print("m:")
        # print(m)
        # print("b:")
        # print(b)
        qq, errs, _, _ = np.linalg.lstsq(m, b, rcond=None)
        # print("residuals:", errs)
        self.q = qq.reshape((self.num_actions, self.feature_size))

    def print(self):
        print("Q:")
        print(self.q)


def main():
    # set up reward matrix for a simple explicit graph
    game = menu.Menu(num_items=5, always_bottom=False)
    frame_shape = game.render(game.sample_scenario()).shape

    num_batch_episodes = 0
    num_episodes = 100
    discount = .9

    #q = GridEstimator(frame_shape, len(actions), learning_rate=1.)
    q = LinearEstimator(frame_shape, len(game.actions), learning_rate=.01)
    q.q = np.random.randn(*q.q.shape) * 1e-3

    # TODO: hold examples in episode buffer
    # TODO: maintain a separate target network

    # Run online episodes (updates on individual trajectories)
    for i in range(num_episodes):
        print("\nONLINE EPISODE %d" % i)
        state = game.sample_scenario()
        frame = game.render(state)
        while True:
            # pick an action
            # TODO: use current policy with probability p
            action = random.randint(0, len(game.actions) - 1)

            # evaluate transition function
            try:
                next_state, reward = game.transition(state, game.actions[action])
            except menu.BadTransition:
                # this action is not valid in this state -- assume no transition and no reward
                next_state = state
                reward = 0.

            # report
            print("  cur_state=%d, action=%s, next_state=%d -> reward = %d" %
                  (state.selection, game.actions[action], -1 if next_state is None else next_state.selection, reward))

            # compute target
            target = reward
            if next_state is not None:
                next_frame = game.render(next_state)
                target += discount * np.max(q.evaluate(next_frame))

            # perform gradient update
            q.update(frame, action, target)

            # if at goal state then terminate
            if next_state is None:
                break

            # move to next state
            state = next_state
            frame = next_frame

        print("Q:")
        for state in game.statespace():
            print("  %s -> %s" % (str(state), str(q.evaluate(game.render(state)))))

    # Run batch episodes (updates on entire state space)
    for i in range(num_batch_episodes):
        print("\nBATCH EPISODE %d" % i)
        frames = []
        actions = []
        targets = []
        for state in game.statespace():
            for action in range(len(game.actions)):
                # evaluate transition function
                try:
                    next_state, reward = game.transition(state, game.actions[action])
                except menu.BadTransition:
                    # this action is not valid in this state -- assume no transition and no reward
                    next_state = state
                    reward = 0.

                # compute target
                target = reward
                if next_state is not None:
                    next_frame = game.render(next_state)
                    target += discount * np.max(q.evaluate(next_frame))

                print("  %s -> %s -> %s -> %.4f" % (str(state), str(next_state), str(q.evaluate(next_frame)), target))

                frames.append(game.render(state))
                actions.append(action)
                targets.append(target)

        q.solve(frames, actions, targets)

        print("Q:")
        for state in game.statespace():
            print("  %s -> %s" % (str(state), str(q.evaluate(game.render(state)))))

    print("\nPOLICY:")
    policy = [["" for _ in range(game.opts.num_items)] for _ in range(game.opts.num_items)]
    for state in game.statespace():
        frame = game.render(state)
        policy[state.selection][state.goal] = game.actions[np.argmax(q.evaluate(frame))]

    print(np.asarray(policy))

    print("\nEVALUATIONS:")
    for state in game.statespace():
        trajectory = []
        outcome = ""
        total_reward = 0.
        for _ in range(20):
            action = np.argmax(q.evaluate(game.render(state)))
            trajectory.append((state, game.actions[action]))
            try:
                next_state, reward = game.transition(state, game.actions[action])
            except menu.BadTransition:
                outcome = "BadTransition"
                break
            total_reward += reward
            if next_state is None:
                outcome = "SUCCESS"
                break
            if next_state == state:
                outcome = "LOOP"
                break
            state = next_state

        #print("%-10s %s" % (str(total_reward), " -> ".join(map(str, trajectory))))
        print("%-10s %s -> %s" % (str(total_reward), game.format_trajectory(trajectory), outcome))


if __name__ == "__main__":
    main()
