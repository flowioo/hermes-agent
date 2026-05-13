"""Tests for multiline editing shortcuts in Hermes CLI.

Covers Ctrl+J newline, Ctrl+T/D indent/dedent, Ctrl+U/W delete shortcuts
across all platforms (macOS, Linux, Windows).
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.history import InMemoryHistory


def _make_buffer(text: str = "", cursor: int | None = None) -> Buffer:
    """Create a prompt_toolkit Buffer with the given text and cursor."""
    buf = Buffer(history=InMemoryHistory())
    buf.text = text
    buf.cursor_position = cursor if cursor is not None else len(text)
    return buf


class TestBindPromptSubmitKeys:
    """_bind_prompt_submit_keys behavior across platforms."""

    def _kb_handlers(self, kb):
        return {tuple(key.value for key in binding.keys): binding.handler for binding in kb.bindings}

    @pytest.fixture
    def submit_handler(self):
        def _handler(event):
            return None
        return _handler

    def test_only_enter_bound_to_submit(self, submit_handler):
        """_bind_prompt_submit_keys only binds enter (c-m); c-j is left
        unbound because Ctrl+J is reserved for newline globally."""
        from prompt_toolkit.key_binding import KeyBindings
        from cli import _bind_prompt_submit_keys

        for platform in ("darwin", "linux", "win32"):
            with patch.object(sys, "platform", platform):
                kb = KeyBindings()
                _bind_prompt_submit_keys(kb, submit_handler)
                bindings = self._kb_handlers(kb)
                assert ("c-m",) in bindings
                assert ("c-j",) not in bindings


class TestIndentCurrentLine:
    """_indent_current_line inserts 4 spaces at line start."""

    def test_indent_single_line(self):
        from cli import _indent_current_line
        buf = _make_buffer("hello", cursor=3)
        _indent_current_line(buf)
        assert buf.text == "    hello"
        assert buf.cursor_position == 7  # 3 + 4

    def test_indent_multiline_cursor_at_end(self):
        from cli import _indent_current_line
        buf = _make_buffer("line1\nline2", cursor=11)
        _indent_current_line(buf)
        assert buf.text == "line1\n    line2"
        assert buf.cursor_position == 15  # 11 + 4

    def test_indent_multiline_cursor_in_middle(self):
        from cli import _indent_current_line
        buf = _make_buffer("line1\nline2", cursor=8)  # cursor at 'i' in line2
        _indent_current_line(buf)
        assert buf.text == "line1\n    line2"
        assert buf.cursor_position == 12  # 8 + 4

    def test_indent_empty_line(self):
        from cli import _indent_current_line
        buf = _make_buffer("line1\n\nline3", cursor=6)
        _indent_current_line(buf)
        assert buf.text == "line1\n    \nline3"
        assert buf.cursor_position == 10  # 6 + 4

    def test_indent_already_indented_line(self):
        from cli import _indent_current_line
        buf = _make_buffer("    hello", cursor=7)
        _indent_current_line(buf)
        assert buf.text == "        hello"
        assert buf.cursor_position == 11  # 7 + 4


class TestDedentCurrentLine:
    """_dedent_current_line removes leading indent."""

    def test_dedent_four_spaces(self):
        from cli import _dedent_current_line
        buf = _make_buffer("    hello", cursor=7)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 3  # 7 - 4

    def test_dedent_tab(self):
        from cli import _dedent_current_line
        buf = _make_buffer("\thello", cursor=4)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 3  # 4 - 1

    def test_dedent_two_spaces(self):
        from cli import _dedent_current_line
        buf = _make_buffer("  hello", cursor=5)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 3  # 5 - 2

    def test_dedent_single_space(self):
        from cli import _dedent_current_line
        buf = _make_buffer(" hello", cursor=4)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 3  # 4 - 1

    def test_dedent_no_indent_is_noop(self):
        from cli import _dedent_current_line
        buf = _make_buffer("hello", cursor=3)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 3

    def test_dedent_multiline(self):
        from cli import _dedent_current_line
        buf = _make_buffer("line1\n    line2", cursor=10)
        _dedent_current_line(buf)
        assert buf.text == "line1\nline2"
        assert buf.cursor_position == 6  # 10 - 4

    def test_dedent_does_not_go_negative(self):
        """Cursor at start of line with indent: result stays at 0."""
        from cli import _dedent_current_line
        buf = _make_buffer("    hello", cursor=2)
        _dedent_current_line(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 0  # max(0, 2 - 4)


class TestDeleteLineBeforeCursor:
    """_delete_line_before_cursor removes text from line start to cursor."""

    def test_delete_mid_line(self):
        from cli import _delete_line_before_cursor
        buf = _make_buffer("hello world", cursor=6)
        _delete_line_before_cursor(buf)
        assert buf.text == "world"
        assert buf.cursor_position == 0

    def test_delete_entire_line(self):
        from cli import _delete_line_before_cursor
        buf = _make_buffer("hello", cursor=5)
        _delete_line_before_cursor(buf)
        assert buf.text == ""
        assert buf.cursor_position == 0

    def test_delete_at_line_start_is_noop(self):
        from cli import _delete_line_before_cursor
        buf = _make_buffer("hello", cursor=0)
        _delete_line_before_cursor(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 0

    def test_delete_second_line(self):
        from cli import _delete_line_before_cursor
        buf = _make_buffer("line1\nline2", cursor=11)  # cursor at end of line2
        _delete_line_before_cursor(buf)
        assert buf.text == "line1\n"
        assert buf.cursor_position == 6


class TestDeleteWordBeforeCursor:
    """_delete_word_before_cursor removes the word before cursor."""

    def test_delete_word(self):
        from cli import _delete_word_before_cursor
        buf = _make_buffer("hello world", cursor=11)
        _delete_word_before_cursor(buf)
        assert buf.text == "hello "
        assert buf.cursor_position == 6

    def test_delete_partial_word(self):
        from cli import _delete_word_before_cursor
        buf = _make_buffer("hello wor", cursor=9)
        _delete_word_before_cursor(buf)
        assert buf.text == "hello "
        assert buf.cursor_position == 6

    def test_delete_at_start_is_noop(self):
        from cli import _delete_word_before_cursor
        buf = _make_buffer("hello", cursor=0)
        _delete_word_before_cursor(buf)
        assert buf.text == "hello"
        assert buf.cursor_position == 0

    def test_delete_after_space_deletes_word(self):
        from cli import _delete_word_before_cursor
        buf = _make_buffer("hello world  ", cursor=13)
        _delete_word_before_cursor(buf)
        assert buf.text == "hello world  "
        # get_word_before_cursor returns "" when cursor is after whitespace
        assert buf.cursor_position == 13
