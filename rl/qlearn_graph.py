import random
import numpy as np

np.set_printoptions(linewidth=300, suppress=True)


def main():
    # set up reward matrix for a simple explicit graph
    rewards = np.array([
        [-1, -1, -1, -1, 0, -1],
        [-1, -1, -1, 0, -1, 100],
        [-1, -1, -1, 0, -1, -1],
        [-1, 0, 0, -1, 0, -1],
        [0, -1, -1, 0, -1, 100],
        [-1, 0, -1, -1, 0, 100]
    ], dtype=float)

    print(rewards)
    print(rewards[4])

    goal_state = 5

    num_states = 6
    num_actions = 6
    num_episodes = 100

    gamma = .8

    q = np.zeros((num_states, num_actions))
    for i in range(num_episodes):
        print("EPISODE %d" % i)
        state = np.random.randint(0, num_states)
        while True:
            # pick an action
            action = random.choice(np.flatnonzero(rewards[state] >= 0))

            # compute transition function
            next_state = action

            # report
            print("  cur_state=%d, action=%d, next_state=%d" %
                  (state, action, next_state))

            # update Q matrix
            q[state, action] = rewards[state, action] + \
                gamma * np.max(q[next_state])

            # if at goal state then terminate
            if next_state == goal_state:
                break

            # move to next state
            state = next_state

    print(q)

    for state in range(num_states):
        seq = [state]
        while state != goal_state:
            next_state = np.argmax(q[state])
            if next_state == state:
                break
            state = next_state
            seq.append(state)
        print(" -> ".join(map(str, seq)))


if __name__ == "__main__":
    main()
