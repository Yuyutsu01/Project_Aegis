import gym
import numpy as np
from gym import spaces


class TradingEnv(gym.Env):
    """
    Trading environment for PPO.
    Compatible with training AND offline inference.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df):
        super().__init__()

        self.data = df.reset_index(drop=True)

        # ===============================
        # FEATURE COLUMNS (MUST MATCH TRAINING)
        # ===============================
        self.feature_cols = [
            "ret_1d_z", "ret_5d_z", "sma_20_z", "sma_50_z",
            "rsi_z", "macd_z", "macd_signal_z",
            "bb_width_z", "vol_sma_z", "vol_ratio_z",
            "trend_sma"
        ]

        self.n_features = len(self.feature_cols)

        # ===============================
        # ACTION & OBSERVATION SPACES
        # ===============================
        self.action_space = spaces.Discrete(3)  # 0: short, 1: flat, 2: long
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_features,),
            dtype=np.float32
        )

        self.current_step = 0
        self.position = 0

        self.reset()

    # ==================================================
    # TRAINING OBSERVATION (USED BY PPO INTERNALLY)
    # ==================================================
    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.feature_cols].values
        return obs.astype(np.float32)

    # ==================================================
    # INFERENCE OBSERVATION (USED BY HARD CONSENSUS)
    # ==================================================
    def get_observation(self, idx: int):
      """
       Deterministic, numerically safe observation for inference."""
   
      obs = self.data.loc[idx, self.feature_cols].values.astype(np.float32)

    # Replace NaN / Inf
      obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clip to training bounds
      obs = np.clip(obs, -10.0, 10.0)

      return obs


    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def step(self, action):
        prev_position = self.position
        self.position = {0: -1, 1: 0, 2: 1}[int(action)]

        ret = self.data.loc[self.current_step, "ret_1d"]
        reward = self.position * ret

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        obs = self._get_obs() if not done else np.zeros(self.n_features, dtype=np.float32)

        return obs, reward, done, False, {}

    def render(self):
        pass
