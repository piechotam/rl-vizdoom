"""Smoke test: verify that ViZDoom initializes correctly with the deathmatch config."""

import sys
from pathlib import Path

import vizdoom as vzd


def main() -> None:
    config_path = (
        Path(__file__).resolve().parent.parent / "scenarios" / "deathmatch.cfg"
    )

    if not config_path.exists():
        print(f"✗ Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    game = vzd.DoomGame()
    game.load_config(str(config_path))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_mode(vzd.Mode.PLAYER)

    game.init()

    game.new_episode()
    num_buttons = game.get_available_buttons_size()

    for _ in range(10):
        if game.is_episode_finished():
            break
        action = [0] * num_buttons
        game.make_action(action)

    game.close()
    print("✓ ViZDoom environment OK")
    print(f"  Config:  {config_path}")
    print(f"  Buttons: {num_buttons}")


if __name__ == "__main__":
    main()
