import gymnasium as gym
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Type


class RewardScaleWrapper(gym.Wrapper):
    """Scale the environment reward."""

    def __init__(self, env, scale):
        """Constructor.

    Args:
      env: A gym env.
      scale: How much to scale the reward by.
    """
        super().__init__(env)

        self._scale = scale

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward *= self._scale
        return obs, reward, done, truncated, info
