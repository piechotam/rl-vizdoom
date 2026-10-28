"""DoomEnv — gym-like wrapper around ViZDoom for the deathmatch scenario."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vizdoom as vzd

# Default config path relative to this file
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "scenarios" / "deathmatch.cfg"
)


class DoomEnv:
    """A thin gym-like wrapper around ViZDoom.

    Exposes ``reset()``, ``step(action)``, and ``close()`` matching the
    standard RL environment interface.

    Action space is represented as one-hot vectors (one button at a time). Input
    is the screen buffer, represented as a ``np.ndarray`` with shape
    ``(H, W, C)``.

    ``info`` contains every game variable declared in the config, keyed by its
    lowercase name (e.g. ``killcount``, ``health``, ``damagecount``), plus
    ``episode_time`` — tics elapsed since the episode started.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        visible: bool = False,
        frame_skip: int = 4,
        seed: int | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._frame_skip = frame_skip

        self._game = vzd.DoomGame()
        self._game.load_config(str(self._config_path))
        self._game.set_window_visible(visible)
        self._game.set_sound_enabled(False)
        if visible:
            self._game.set_mode(vzd.Mode.ASYNC_PLAYER)
            self._game.set_ticrate(60)
        else:
            self._game.set_mode(vzd.Mode.PLAYER)
        if seed is not None:
            self._game.set_seed(seed)
        self._game.init()

        self._game_variables = list(self._game.get_available_game_variables())

        n_buttons = self._game.get_available_buttons_size()
        self._action_space: list[list[int]] = []
        # no-op
        self._action_space.append([0] * n_buttons)
        # one-hot actions
        for i in range(n_buttons):
            action = [0] * n_buttons
            action[i] = 1
            self._action_space.append(action)

    def seed(self, seed: int) -> None:
        """Seed the game engine; takes effect from the next ``reset()``."""
        self._game.set_seed(seed)

    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial observation."""
        self._game.new_episode()
        return self._get_observation()

    def step(self, action_index: int) -> tuple[np.ndarray | None, float, bool, dict]:
        """Execute an action and return ``(obs, reward, done, info)``."""
        action = self._action_space[action_index]
        reward = self._game.make_action(action, self._frame_skip)

        done = self._game.is_episode_finished()

        if done:
            return None, reward, True, self._get_info()

        return self._get_observation(), reward, False, self._get_info()

    def get_info(self) -> dict:
        """Current game variables (also valid right after ``reset()``)."""
        return self._get_info()

    def close(self) -> None:
        """Shut down the ViZDoom instance."""
        self._game.close()

    @property
    def num_actions(self) -> int:
        """Total number of discrete actions (one-hot buttons + no-op)."""
        return len(self._action_space)

    @property
    def action_space(self) -> list[list[int]]:
        """List of all possible actions as button vectors."""
        return self._action_space

    @property
    def frame_skip(self) -> int:
        return self._frame_skip

    @property
    def episode_timeout(self) -> int:
        """Episode length limit in tics."""
        return self._game.get_episode_timeout()

    @property
    def game(self) -> vzd.DoomGame:
        """Access to the underlying ``DoomGame``."""
        return self._game

    def _get_observation(self) -> np.ndarray:
        state = self._game.get_state()
        return np.transpose(state.screen_buffer, (1, 2, 0))

    def _get_info(self) -> dict:
        state = self._game.get_state()
        if state is not None and state.game_variables is not None:
            values = list(state.game_variables)
        else:
            # Episode finished — no state, read last known values directly
            values = [self._game.get_game_variable(gv) for gv in self._game_variables]
        info = {
            gv.name.lower(): float(value)
            for gv, value in zip(self._game_variables, values, strict=True)
        }
        info["episode_time"] = self._game.get_episode_time()
        return info
