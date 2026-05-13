"""Regression tests for issue #22379 — Ctrl+Enter newline over SSH/WSL.

Enhanced terminals (Kitty, xterm with modifyOtherKeys, mintty) send
Ctrl+Enter as distinct CSI-u sequences. install_ctrl_enter_alias() maps
these to Alt+Enter so the newline handler fires.
"""

from __future__ import annotations


def test_install_ctrl_enter_alias_maps_csi_u_sequences():
    """Kitty / xterm modifyOtherKeys / mintty Ctrl+Enter sequences alias to
    Alt+Enter (Escape, ControlM) so the existing newline handler fires."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
    from prompt_toolkit.keys import Keys

    install_ctrl_enter_alias()
    alt_enter = (Keys.Escape, Keys.ControlM)
    for seq in ("\x1b[13;5u", "\x1b[27;5;13~", "\x1b[27;5;13u"):
        assert ANSI_SEQUENCES.get(seq) == alt_enter, (
            f"Ctrl+Enter sequence {seq!r} not mapped to Alt+Enter tuple"
        )


def test_install_ctrl_enter_alias_idempotent():
    """Running it twice doesn't double-count or break."""
    from hermes_cli.pt_input_extras import install_ctrl_enter_alias
    install_ctrl_enter_alias()
    second = install_ctrl_enter_alias()
    assert second == 0  # no further changes after first install
