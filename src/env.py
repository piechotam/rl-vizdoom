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
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        visible: bool = False,
        frame_skip: int = 4,
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
        self._game.init()

        n_buttons = self._game.get_available_buttons_size()
        self._action_space: list[list[int]] = []
        # no-op
        self._action_space.append([0] * n_buttons)
        # one-hot actions
        for i in range(n_buttons):
            action = [0] * n_buttons
            action[i] = 1
            self._action_space.append(action)

    def reset(self) -> np.ndarray:
        """Start a new episode and return the initial observation."""
        self._game.new_episode()
        return self._get_observation()

    def step(
        self, action_index: int
    ) -> tuple[np.ndarray | None, float, bool, dict]:
        """Execute an action and return ``(obs, reward, done, info)``."""
        action = self._action_space[action_index]
        reward = self._game.make_action(action, self._frame_skip)

        done = self._game.is_episode_finished()

        if done:
            return None, reward, True, self._get_info()

        return self._get_observation(), reward, False, self._get_info()

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
    def game(self) -> vzd.DoomGame:
        """Access to the underlying ``DoomGame``."""
        return self._game

    def _get_observation(self) -> np.ndarray:
        state = self._game.get_state()
        return np.transpose(state.screen_buffer, (1, 2, 0))

    def _get_info(self) -> dict:
        gv = (
            self._game.get_state().game_variables
            if not self._game.is_episode_finished()
            else None
        )
        if gv is not None:
            return {
                "kill_count": int(gv[0]),
                "health": float(gv[1]),
                "armor": float(gv[2]),
                "selected_weapon": int(gv[3]),
                "ammo": int(gv[4]),
            }
        # Episode finished — read last known variables via the game object
        return {
            "kill_count": self._game.get_game_variable(
                vzd.GameVariable.KILLCOUNT
            ),
            "health": self._game.get_game_variable(vzd.GameVariable.HEALTH),
            "armor": self._game.get_game_variable(vzd.GameVariable.ARMOR),
            "selected_weapon": self._game.get_game_variable(
                vzd.GameVariable.SELECTED_WEAPON
            ),
            "ammo": self._game.get_game_variable(
                vzd.GameVariable.SELECTED_WEAPON_AMMO
            ),
        }
