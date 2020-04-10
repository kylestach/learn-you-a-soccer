from gym.envs.registration import register

register(
      id='robocup-collect-v0',
      entry_point='robocup_env.envs:RoboCup',
)
