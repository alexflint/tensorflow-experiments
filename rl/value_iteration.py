import numpy as np


class BadTransition(Exception):
    pass


class GridWorld(object):
    def __init__(self, rows):
        self.rows = rows
        self.nc = max(len(row) for row in rows)
        self.nr = len(rows)

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1), "exit"]
        self.states = []
        self.begin = None
        for i, row in enumerate(rows):
            for j, ch in enumerate(row):
                if ch != "#":
                    self.states.append((i, j))
                if ch == "B":
                    self.begin = (i, j)

    def transition(self, state, action):
        ch = self.rows[state[0]][state[1]]
        if (ch == "G" or ch == "P") and action != "exit":
            raise BadTransition()

        if action == "exit":
            if ch == "G":
                return None, 1.
            if ch == "P":
                return None, -1.
            raise BadTransition()

        ii, jj = state[0] + action[0], state[1] + action[1]
        if ii < 0 or ii >= self.nr or jj < 0 or jj >= self.nc:
            raise BadTransition()
        if self.rows[ii][jj] == "#":
            raise BadTransition()
        return (ii, jj), 0.

    def actions_from(self, state):
        actions = []
        for a in self.actions[:4]:
            try:
                self.transition(state, a)
                actions.append(a)
            except BadTransition:
                pass
        return actions


def main():
    mdp = GridWorld([
        "   G",
        " # P",
        "B   ",
    ])

    horizon = 10
    discount = .9

    vs = []
    ps = []
    v_prev = np.zeros(len(mdp.states))
    for h in range(horizon):
        #print("\n\niteration {}".format(h))
        v_cur = []
        p_cur = []
        for i, state in enumerate(mdp.states):
            best_value = None
            best_action = None
            for a in mdp.actions:
                try:
                    next_state, reward = mdp.transition(state, a)

                    cur = reward
                    if next_state is not None:
                        cur += discount * v_prev[mdp.states.index(next_state)]

                    if best_value is None or cur > best_value:
                        best_value = cur
                        best_action = a
                except BadTransition:
                    pass

            v_cur.append(best_value)
            p_cur.append(best_action)

        v_prev = np.asarray(v_cur)
        vs.append(v_cur)
        ps.append(p_cur)

    for state in mdp.states:
        trajectory = []
        total_reward = 0
        while state is not None:
            trajectory.append(state)
            state, reward = mdp.transition(
                state, p_cur[mdp.states.index(state)])
            total_reward += reward

        print("{}    -> {}".format(total_reward, trajectory))


if __name__ == "__main__":
    main()
