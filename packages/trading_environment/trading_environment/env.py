import gym
import numpy as np
from gym import spaces


class TradingEnv(gym.Env):
    """
    Trading environment for PPO (Meta-Policy).
    Uses precomputed XGBoost probabilities and current position as part of observation.
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

        if "xgb_prob" not in self.data.columns:
            raise ValueError("Data must contain 'xgb_prob' column for Meta-Policy")

        # Observation: [Features...] + [xgb_prob] + [position]
        self.n_features = len(self.feature_cols) + 2

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
        self.commission = 0.001  # 0.1% per side (fixed)

        self.reset()

    # ==================================================
    # TRAINING OBSERVATION
    # ==================================================
    def _get_obs(self):
        obs = self.data.loc[self.current_step, self.feature_cols].values.tolist()
        xgb_prob = self.data.loc[self.current_step, "xgb_prob"]
        obs.append(xgb_prob)
        obs.append(self.position)
        return np.array(obs, dtype=np.float32)

    # ==================================================
    # INFERENCE OBSERVATION (USED BY HARD CONSENSUS)
    # ==================================================
    def get_observation(self, idx: int, current_position: int, xgb_prob: float):
      """
       Deterministic, numerically safe observation for inference."""
   
      obs = self.data.loc[idx, self.feature_cols].values.tolist()
      obs.append(xgb_prob)
      obs.append(current_position)
      obs_arr = np.array(obs, dtype=np.float32)

    # Replace NaN / Inf
      obs_arr = np.nan_to_num(obs_arr, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clip to training bounds
      obs_arr = np.clip(obs_arr, -10.0, 10.0)

      return obs_arr


    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.position = 0
        return self._get_obs(), {}

    def step(self, action):
        prev_position = self.position
        new_position = {0: -1, 1: 0, 2: 1}[int(action)]
        self.position = new_position

        # --------------------------------------------------
        # TRADING COST (Commission + Slippage)
        # Applied only when position changes
        # --------------------------------------------------
        costs = 0
        if new_position != prev_position:
            costs = self.commission

        # --------------------------------------------------
        # REWARD DELAY (NO LOOK-AHEAD)
        # Observation at t -> Action at t -> earns return at t+1
        # --------------------------------------------------
        next_ret = self.data.loc[self.current_step + 1, "ret_1d"]
        
        # Meta-Policy Reward: Penalize excessive switching and optimize for Sharpe-like behavior
        step_pnl = (self.position * next_ret) - costs
        
        # We give a slight penalty for being in cash to encourage finding good trades, 
        # or we could just use step_pnl. Let's use pure step_pnl.
        reward = step_pnl

        self.current_step += 1
        done = self.current_step >= len(self.data) - 2 # buffer for t+1 return

        obs = self._get_obs() if not done else np.zeros(self.n_features, dtype=np.float32)

        return obs, reward, done, False, {}

    def render(self):
        pass
