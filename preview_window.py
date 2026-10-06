"""Preview window helpers. Display only — never sends input.

The OpenCV window is a normal overlapped window. Windows decides which
window is in front when the user clicks. This module does not change
extended styles, does not raise the window, and does not call
SetForegroundWindow. UI shortcuts are gated separately by window_focus.

Frames are cover-cropped to the real client/fullscreen size so the image
fills the window: no gray, black, or empty bands.
"""
import os

import cv2
import numpy as np


WINDOW_FLAGS = cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO


def hide_preview(title):
    """Close the OpenCV preview. Recognition must keep running in the caller."""
    try:
        cv2.destroyWindow(title)
    except cv2.error:
        pass


def ensure_preview_window(title):
    """Create a user-resizable window that may differ in aspect from the camera."""
    try:
        visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        visible = -1.0
    if visible < 1:
        cv2.namedWindow(title, WINDOW_FLAGS)
    try:
        cv2.setWindowProperty(title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    except cv2.error:
        pass


def preview_was_closed(title):
    """True after the user clicks the window X. Hidden-preview is handled by the caller."""
    try:
        value = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return True
    return value < 1


def screen_size():
    """Primary display size. Read-only; never changes z-order or focus."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        width = int(ctypes.windll.user32.GetSystemMetrics(0))
        height = int(ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        return None
    if width < 16 or height < 16:
        return None
    return width, height


def preview_client_size(title, frame):
    """Client size to fill. Fullscreen uses the screen when OpenCV reports a small rect."""
    fallback = (int(frame.shape[1]), int(frame.shape[0]))
    try:
        fullscreen = cv2.getWindowProperty(title, cv2.WND_PROP_FULLSCREEN)
    except cv2.error:
        fullscreen = 0
    width = height = 0
    try:
        _x, _y, width, height = cv2.getWindowImageRect(title)
    except cv2.error:
        width = height = 0
    width, height = int(width or 0), int(height or 0)
    if fullscreen > 0:
        screen = screen_size()
        if screen is not None:
            sw, sh = screen
            if width < int(sw * 0.9) or height < int(sh * 0.9):
                return sw, sh
            return max(width, sw), max(height, sh)
    if width < 16 or height < 16:
        return fallback
    return width, height


def cover_resize(frame, dest_width, dest_height):
    """Scale to cover dest, then crop. Result is exactly dest_width x dest_height."""
    dest_width = int(dest_width)
    dest_height = int(dest_height)
    if dest_width < 1 or dest_height < 1 or frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    if width == dest_width and height == dest_height:
        return frame
    scale = max(dest_width / float(width), dest_height / float(height))
    new_width = max(dest_width, int(round(width * scale)))
    new_height = max(dest_height, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    x = max(0, (new_width - dest_width) // 2)
    y = max(0, (new_height - dest_height) // 2)
    cropped = resized[y:y + dest_height, x:x + dest_width]
    if cropped.shape[1] != dest_width or cropped.shape[0] != dest_height:
        cropped = cv2.resize(cropped, (dest_width, dest_height), interpolation=interpolation)
    return np.ascontiguousarray(cropped)


def fit_preview_frame(title, frame):
    """One resize/crop so the numpy image matches the live window or fullscreen surface."""
    ensure_preview_window(title)
    dest_width, dest_height = preview_client_size(title, frame)
    return cover_resize(frame, dest_width, dest_height)


def sidebar_width_for(dest_width):
    """Fixed-ish left column (about 250-300 px). Camera keeps the rest."""
    dest_width = int(dest_width)
    if dest_width < 500:
        return max(160, min(220, int(dest_width * 0.40)))
    return max(250, min(300, int(dest_width * 0.24)))


def compose_sidebar_camera(camera, dest_width, dest_height, sidebar_w=None):
    """Sidebar | cover-cropped camera. The canvas is exactly dest size: no gray bands."""
    dest_width = int(dest_width)
    dest_height = int(dest_height)
    if dest_width < 16 or dest_height < 16 or camera is None or camera.size == 0:
        return camera, 0
    if sidebar_w is None:
        sidebar_w = sidebar_width_for(dest_width)
    sidebar_w = max(0, min(int(sidebar_w), dest_width - 16))
    canvas = np.zeros((dest_height, dest_width, 3), dtype=np.uint8)
    canvas[:] = (28, 26, 24)
    cam_w = dest_width - sidebar_w
    if cam_w >= 1:
        pane = cover_resize(camera, cam_w, dest_height)
        canvas[:, sidebar_w:] = pane
    return np.ascontiguousarray(canvas), sidebar_w


def apply_window_icon(title):
    """Set the ICO on our OpenCV window. Does not raise or activate it."""
    if os.name != "nt" or not title:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        hwnd = user32.FindWindowW(None, str(title))
        if not hwnd:
            return False
        from config import resource_path
        path = resource_path(os.path.join("assets", "handcontroller_icon.ico"))
        if not os.path.isfile(path):
            return False
        IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON = 1, 0x0010, 0x0080
        ICON_SMALL, ICON_BIG = 0, 1
        SM_CXICON, SM_CYICON, SM_CXSMICON, SM_CYSMICON = 11, 12, 49, 50
        user32.LoadImageW.restype = wintypes.HANDLE
        user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                      ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        # One handle per size: Windows picks the matching ICO entry instead of
        # stretching the first (16 px) image into the taskbar / Alt+Tab icon.
        small = user32.LoadImageW(None, path, IMAGE_ICON, user32.GetSystemMetrics(SM_CXSMICON),
                                  user32.GetSystemMetrics(SM_CYSMICON), LR_LOADFROMFILE)
        big = user32.LoadImageW(None, path, IMAGE_ICON, user32.GetSystemMetrics(SM_CXICON),
                                user32.GetSystemMetrics(SM_CYICON), LR_LOADFROMFILE)
        if not small and not big:
            return False
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small or big)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big or small)
        return True
    except Exception:
        return False


def bind_preview_mouse(title, callback):
    """Click handler on our preview. Does not raise or activate any window."""
    ensure_preview_window(title)
    try:
        cv2.setMouseCallback(title, callback)
    except cv2.error:
        pass


def show_preview_frame(title, frame):
    """Draw one preview frame. Does not change which window is in front."""
    ensure_preview_window(title)
    cv2.imshow(title, frame)
