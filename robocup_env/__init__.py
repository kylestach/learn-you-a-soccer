from gym.envs.registration import register

register(
      id='robocup-collect-v0',
      entry_point='robocup_env.envs:RoboCupCollect',
)
register(
      id='robocup-score-v0',
      entry_point='robocup_env.envs:RoboCupScore',
)
register(
      id='robocup-pass-v0',
      entry_point='robocup_env.envs:RoboCupPass',
)
