"""DoomEnv — gym-like wrapper around ViZDoom for the deathmatch scenario."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vizdoom as vzd

from src.actions import ACTION_NAMES, ACTIONS, build_button_vectors
from src.game_state import enemy_flags

# Default config path relative to this file
_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "scenarios" / "deathmatch.cfg"
)


class DoomEnv:
    """A thin gym-like wrapper around ViZDoom.

    Exposes ``reset()``, ``step(action)``, and ``close()`` matching the
    standard RL environment interface.

    The action space is the fixed list of 31 button combinations from
    ``src.actions.ACTIONS``. Input is the screen buffer, represented as a
    ``np.ndarray`` with shape ``(H, W, C)``.

    ``info`` contains every game variable declared in the config, keyed by its
    lowercase name (e.g. ``killcount``, ``health``, ``damagecount``), plus
    ``episode_time`` — tics elapsed since the episode started. With
    ``labels=True`` it also has ``enemy_visible`` and ``enemy_in_crosshair``
    (0/1, from the labels buffer; see ``src.game_state``).
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        visible: bool = False,
        frame_skip: int = 4,
        seed: int | None = None,
        skill: int | None = None,
        labels: bool = False,
    ) -> None:
        self._config_path = Path(config_path)
        self._frame_skip = frame_skip
        self._labels = labels

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
        if skill is not None:
            self._game.set_doom_skill(skill)
        self._game.set_labels_buffer_enabled(labels)
        self._game.init()

        self._game_variables = list(self._game.get_available_game_variables())

        self._action_space = build_button_vectors(
            list(self._game.get_available_buttons())
        )

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
        """Total number of discrete actions."""
        return len(self._action_space)

    @property
    def action_space(self) -> list[list[int]]:
        """List of all possible actions as button vectors."""
        return self._action_space

    @property
    def action_names(self) -> tuple[str, ...]:
        """Labels of the actions, e.g. ``"TURN_LEFT+ATTACK"``."""
        return ACTION_NAMES

    def actions_with(self, button: vzd.Button) -> list[int]:
        """Indices of every action that presses ``button``."""
        return [i for i, action in enumerate(ACTIONS) if button.name in action]

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
        if self._labels:
            labels = (state.labels if state is not None else None) or []
            info |= enemy_flags(
                labels,
                self._game.get_screen_width(),
                self._game.get_screen_height(),
            )
        return info
