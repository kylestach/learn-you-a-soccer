from pyglet.window import key
import numpy as np
import gym


def handle_key_press(k, restart, force):
    if k == key.SPACE:
        restart = True
    # Robot 1
    if k == key.UP:
        force[1] = 1
    if k == key.DOWN:
        force[1] = -1
    if k == key.LEFT:
        force[0] = -1
    if k == key.RIGHT:
        force[0] = 1
    if k == key.APOSTROPHE:
        force[2] = -1
    if k == key.L:
        force[2] = 1
    if k == key.P:
        force[3] = 1

    # Robot 2
    if k == key.T:
        force[4 + 1] = 1
    if k == key.G:
        force[4 + 1] = -1
    if k == key.F:
        force[4 + 0] = -1
    if k == key.H:
        force[4 + 0] = 1
    if k == key.D:
        force[4 + 2] = -1
    if k == key.A:
        force[4 + 2] = 1
    if k == key.W:
        force[4 + 3] = 1
    return restart, force


def handle_key_release(k, force):
    # Robot 1
    if k == key.UP:
        force[1] = 0
    if k == key.DOWN:
        force[1] = 0
    if k == key.LEFT:
        force[0] = 0
    if k == key.RIGHT:
        force[0] = 0
    if k == key.APOSTROPHE:
        force[2] = 0
    if k == key.L:
        force[2] = 0
    if k == key.P:
        force[3] = 0

    # Robot 2
    if k == key.T:
        force[4 + 1] = 0
    if k == key.G:
        force[4 + 1] = 0
    if k == key.F:
        force[4 + 0] = 0
    if k == key.H:
        force[4 + 0] = 0
    if k == key.D:
        force[4 + 2] = 0
    if k == key.A:
        force[4 + 2] = 0
    if k == key.W:
        force[4 + 3] = 0
    return force


def main():
    restart = False
    robocup_env_name = "pass"

    if robocup_env_name == "collect":
        force = np.zeros(4)
    elif robocup_env_name == "collect":
        force = np.zeros(4)
    elif robocup_env_name == "pass":
        force = np.zeros(8)

    env = gym.make(f"robocup_env:robocup-{robocup_env_name}-v0")

    def key_press(k, mod):
        nonlocal restart, force
        restart, force = handle_key_press(k, restart, force)

    def key_release(k, mod):
        nonlocal force
        force = handle_key_release(k, force)

    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    is_open = True
    np.set_printoptions(2, linewidth=150, floatmode="fixed", sign=" ", suppress=True)

    while is_open:
        env.reset()
        restart = False
        total_r = 0
        while True:
            s, r, done, info = env.step(force)
            # print(f"env.robot.angle: {env.robot.angle:3f}    state: {s[2]:3f}, {s[3]:3f}    can_kick: {s[4]}")
            # print(s)
            total_r += r
            is_open = env.render()
            if done or restart:
                print("total r: ", total_r)
                break


if __name__ == '__main__':
    main()
