# cell 1: imports and basic config
import math
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# For stable-baselines3 later
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt


# cell 2: the custom environment class (Gymnasium API)
class SingleProjectBudgetEnv(gym.Env):
    """
    Single Project Budget Allocation environment (Gymnasium API).

    Observation:
        Type: Box(3)
        [remaining_budget_frac, timesteps_left_frac, cumulative_spent_frac]
        All in [0,1]

    Actions:
        Type: Discrete(4) -> map to fractions of *remaining budget*:
            0 -> 0.0
            1 -> 0.25
            2 -> 0.5
            3 -> 1.0

    Rewards:
        +1 for progress (allocated > 0 and not overspend)
        -10 for overspend (defensive)
        +100 for completion (cumulative_spent >= completion_threshold)
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 total_budget: float = 100.0,
                 max_timesteps: int = 10,
                 completion_threshold: float = 90.0,
                 render_mode: str | None = None):
        super().__init__()

        self.total_budget = float(total_budget)
        self.max_timesteps = int(max_timesteps)
        self.completion_threshold = float(completion_threshold)

        # Action mapping (fractions of remaining budget)
        self.action_map = [0.0, 0.25, 0.5, 1.0]

        # Define action & observation spaces
        self.action_space = spaces.Discrete(len(self.action_map))
        # obs: remaining_frac, timesteps_left_frac, cumulative_spent_frac
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

        # internal state
        self.current_step = None
        self.remaining_budget = None
        self.cumulative_spent = None
        self.done = None
        self.render_mode = render_mode

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Gymnasium seeding pattern
        super().reset(seed=seed)
        self.current_step = 0
        self.remaining_budget = float(self.total_budget)
        self.cumulative_spent = 0.0
        self.done = False

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        return np.array([
            self.remaining_budget / self.total_budget,
            (self.max_timesteps - self.current_step) / max(1, self.max_timesteps),
            self.cumulative_spent / self.total_budget
        ], dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        frac = self.action_map[int(action)]
        allocated = frac * self.remaining_budget  # allocation is fraction of *current remaining* budget

        reward = 0.0
        info = {}

        # Overspend defensive check (shouldn't happen with our action map)
        if allocated > self.remaining_budget + 1e-8:
            # overspend
            reward += -10.0
            self.done = True
            terminated = True
            truncated = False
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # Apply allocation
        self.remaining_budget -= allocated
        self.cumulative_spent += allocated

        # Progress reward: +1 for making progress this step (allocated > 0)
        if allocated > 0:
            reward += 1.0

        # Completion check
        if self.cumulative_spent >= self.completion_threshold:
            reward += 100.0
            self.done = True
            terminated = True
            truncated = False
            obs = self._get_obs()
            return obs, reward, terminated, truncated, info

        # Step advancement
        self.current_step += 1
        # If reached max timesteps -> truncated episode
        if self.current_step >= self.max_timesteps:
            self.done = True
            terminated = False
            truncated = True
        else:
            terminated = False
            truncated = False

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.current_step}: remaining={self.remaining_budget:.2f}, "
                  f"spent={self.cumulative_spent:.2f}")
        else:
            return None

    def close(self):
        pass


# cell: sanity check
env = SingleProjectBudgetEnv(total_budget=100, max_timesteps=10, completion_threshold=90.0, render_mode="human")
obs, info = env.reset()
print("Initial obs:", obs)
# try a greedy policy: always 25% of remaining
for _ in range(5):
    obs, r, term, trunc, info = env.step(1)  # action 1 -> 25%
    env.render()
    print("reward", r, "term", term, "trunc", trunc)
    if term or trunc:
        break

# cell: create vectorized & normalized envs and model
log_dir = "./logs_sb3/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank=0):
    def _init():
        env = SingleProjectBudgetEnv(total_budget=100, max_timesteps=10, completion_threshold=90.0)
        # Monitor writes episode info to csv for analysis if filename provided
        return Monitor(env)
    return _init

n_envs = 4
venv = DummyVecEnv([make_env(i) for i in range(n_envs)])
# Normalize observations (not rewards here). Stable Baselines3 has VecNormalize.
venv = VecNormalize(venv, norm_obs=True, norm_reward=False)


# cell: create model
model = PPO(
    policy="MlpPolicy",
    env=venv,
    verbose=1,
    tensorboard_log="./tb_logs/",
    learning_rate=3e-4,
    n_steps=2048,     # amount of experience per environment per update
    batch_size=64,
    ent_coef=0.0,
    clip_range=0.2    # important PPO hyperparameter (explained below)
)


# cell: train
total_timesteps = 50_000  # small for a toy env; increase to 200k+ for sturdier learning
model.learn(total_timesteps=total_timesteps)
# save model
model.save("ppo_budget_agent")
