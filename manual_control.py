from robocup_env.envs.robocup import RoboCup
from pyglet.window import key
import numpy as np
import gym


def handle_key_press(k, restart, force):
    if k == key.SPACE:
        restart = True
    if k == key.UP:
        force[1] = 1
    if k == key.DOWN:
        force[1] = -1
    if k == key.LEFT:
        force[0] = -1
    if k == key.RIGHT:
        force[0] = 1
    if k == key.A:
        force[2] = -1
    if k == key.D:
        force[2] = 1
    if k == key.W:
        force[3] = 1
    return restart, force


def handle_key_release(k, force):
    if k == key.UP:
        force[1] = 0
    if k == key.DOWN:
        force[1] = 0
    if k == key.LEFT:
        force[0] = 0
    if k == key.RIGHT:
        force[0] = 0
    if k == key.A:
        force[2] = 0
    if k == key.D:
        force[2] = 0
    if k == key.W:
        force[3] = 0
    return force


def main():
    restart = False
    env = gym.make("robocup_env:robocup-collect-v0")

    force = np.array([0.0, 0.0, 0.0, 0.0])

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
    while is_open:
        env.reset()
        restart = False
        total_r = 0
        while True:
            s, r, done, info = env.step(force)
            # print(f"env.robot.angle: {env.robot.angle:3f}    state: {s[2]:3f}, {s[3]:3f}    can_kick: {s[4]}")
            total_r += r
            # print("total r: ", total_r)
            is_open = env.render()
            if done or restart:
                break


if __name__ == '__main__':
    main()
