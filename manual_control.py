from robocup import RoboCup
from pyglet.window import key


def handle_key_press(k, restart, force):
    if k == key.SPACE:
        restart = True
    if k == key.UP:
        force[1] = 10
    if k == key.DOWN:
        force[1] = -10
    if k == key.LEFT:
        force[0] = -10
    if k == key.RIGHT:
        force[0] = 10
    if k == key.A:
        force[2] = -80
    if k == key.D:
        force[2] = 80
    if k == key.W:
        force[3] = 2.5e-3
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
    env = RoboCup()

    force = [0.0, 0.0, 0.0, 0.0]

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
        while True:
            s, r, done, info = env.step(force)
            is_open = env.render()
            if done or restart:
                break


if __name__ == '__main__':
    main()
