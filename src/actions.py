"""Discrete action space: 31 button combinations used by every agent.

Each action is a tuple of button names pressed together for ``frame_skip`` tics.
``SPEED`` is a modifier that doubles both movement and turning speed (measured:
14.1 deg/step turning without it vs 28.1 deg/step with it at ``frame_skip=4``).
"""

from __future__ import annotations

from collections.abc import Sequence

import vizdoom as vzd

ACTIONS: tuple[tuple[str, ...], ...] = (
    # basic
    (),
    ("ATTACK",),
    ("MOVE_FORWARD",),
    ("MOVE_BACKWARD",),
    ("MOVE_LEFT",),
    ("MOVE_RIGHT",),
    ("TURN_LEFT",),
    ("TURN_RIGHT",),
    # weapon
    ("SELECT_NEXT_WEAPON",),
    ("SELECT_PREV_WEAPON",),
    # attack combos
    ("TURN_LEFT", "ATTACK"),
    ("TURN_RIGHT", "ATTACK"),
    ("MOVE_FORWARD", "ATTACK"),
    ("MOVE_BACKWARD", "ATTACK"),
    ("MOVE_LEFT", "ATTACK"),
    ("MOVE_RIGHT", "ATTACK"),
    # movement combos
    ("MOVE_FORWARD", "TURN_LEFT"),
    ("MOVE_FORWARD", "TURN_RIGHT"),
    ("MOVE_LEFT", "TURN_RIGHT"),  # circle strafe
    ("MOVE_RIGHT", "TURN_LEFT"),  # circle strafe
    # sprint
    ("SPEED", "MOVE_FORWARD"),
    ("SPEED", "MOVE_BACKWARD"),
    ("SPEED", "MOVE_LEFT"),
    ("SPEED", "MOVE_RIGHT"),
    ("SPEED", "TURN_LEFT"),
    ("SPEED", "TURN_RIGHT"),
    # sprint triples
    ("SPEED", "MOVE_FORWARD", "TURN_LEFT"),
    ("SPEED", "MOVE_FORWARD", "TURN_RIGHT"),
    ("SPEED", "MOVE_FORWARD", "ATTACK"),
    ("SPEED", "MOVE_LEFT", "ATTACK"),
    ("SPEED", "MOVE_RIGHT", "ATTACK"),
)


def action_name(action: tuple[str, ...]) -> str:
    """Human-readable label, e.g. ``"SPEED+MOVE_FORWARD"`` or ``"NOOP"``."""
    return "+".join(action) if action else "NOOP"


ACTION_NAMES: tuple[str, ...] = tuple(action_name(a) for a in ACTIONS)


def build_button_vectors(
    buttons: Sequence[vzd.Button],
    actions: Sequence[tuple[str, ...]] = ACTIONS,
) -> list[list[int]]:
    """Turn button-name combinations into binary vectors for ``make_action``.

    ``buttons`` is the order returned by ``DoomGame.get_available_buttons()``.
    Raises ``ValueError`` if an action uses a button that is not available.
    """
    index = {b.name: i for i, b in enumerate(buttons)}
    vectors = []
    for action in actions:
        vector = [0] * len(buttons)
        for name in action:
            if name not in index:
                raise ValueError(f"Button {name} is not in the scenario config")
            vector[index[name]] = 1
        vectors.append(vector)
    return vectors
