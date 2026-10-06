"""Whether the HandTrack preview is the foreground window.

This does not activate, raise, or target any other window.
"""
import ctypes
import os


def interface_keys_enabled(focused):
    """Preview UI clicks and leftover waitKey handlers only while HandController is in front."""
    return bool(focused)


def own_window_is_foreground(title):
    """True when the window titled exactly `title` is in front.

    Exact match: the console (...\\HandController.exe) or an editor showing a
    HandController file must not count as the preview window.

    On non-Windows platforms the check is unavailable, so the UI keys stay
    enabled. Gesture processing never consults this function.
    """
    if os.name != "nt" or not title:
        return True
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip().lower() == title.strip().lower()
