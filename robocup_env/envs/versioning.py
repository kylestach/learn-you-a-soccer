from .collect import CollectConfig
from .score import ScoreConfig
from .passing import PassConfig


class NegativeScoreConfig(ScoreConfig):
    def __init__(self):
        ScoreConfig.__init__(self,
                             dribble_count_done=100,  #
                             dribbling_reward=1.0,  #
                             done_reward_additive=0.0,  #
                             done_reward_coeff=500.0,  #
                             done_reward_exp_base=0.998,  #
                             ball_out_of_bounds_reward=-100.0,  #
                             distance_to_ball_coeff=-0.1,  #
                             survival_reward=0.0,  # 0 survival reward
                             enable_dribble_reward=True,  #
                             enable_distance_reward=True  #
                             )
