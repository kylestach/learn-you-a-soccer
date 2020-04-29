from gym.envs.registration import register
import robocup_env.envs as rc_envs

register(
    id='robocup-collect-v0',
    entry_point='robocup_env.envs:RoboCupCollect',
)

register(
    id='robocup-score-v0',
    entry_point='robocup_env.envs:RoboCupScore',
    kwargs={'score_config': rc_envs.NegativeScoreConfig()}
)

register(
    id='robocup-score-v1',
    entry_point='robocup_env.envs:RoboCupScore'
)

register(
    id='robocup-pass-v0',
    entry_point='robocup_env.envs:RoboCupPass',
)
