"""External click-through OSD. Win32 layered window, never Qt.

Drawn on the primary desktop, top-left, above a focused game when the
compositor allows it. Exclusive fullscreen may hide it. This module never
injects into a game, never calls SetForegroundWindow, and never activates.
The window is created only when shown. Hidden overlay keeps recognition running.
"""
from __future__ import annotations

import os
import sys

OSD_CLASS = "HandControllerOSD"
OSD_TITLE = "HandController OSD"
OSD_X = 12
OSD_Y = 12
OSD_W = 420
OSD_H = 168

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_EX_TOPMOST = 0x00000008
SWP_NOACTIVATE = 0x0010
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
LWA_COLORKEY = 0x00000001
WM_DESTROY = 0x0002
WM_CLOSE = 0x0010
WM_PAINT = 0x000F
WM_ERASEBKGND = 0x0014
WM_NCHITTEST = 0x0084
HTTRANSPARENT = -1
DT_LEFT = 0x0000
DT_NOPREFIX = 0x00000800
DT_SINGLELINE = 0x00000020
DT_END_ELLIPSIS = 0x00008000
TRANSPARENT_BG = 1
COLOR_KEY = 0x00000000
WHITE = 0x00FFFFFF
CYAN = 0x00FFE000
GRAY = 0x00B4B4B4
OUTLINE = 0x00101010  # near-black, distinct from the colour key
NONANTIALIASED_QUALITY = 3
BLACK_BRUSH = 4
ERROR_CLASS_ALREADY_EXISTS = 1410

_instances = {}


def _win32():
    return os.name == "nt" and sys.platform.startswith("win")


def osd_ex_styles():
    """Public flags for tests: click-through, no activate, tool, layered."""
    return WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST


def _hwnd_topmost():
    import ctypes
    return ctypes.c_void_p(-1)


class GameOSD:
    """One top-left overlay. Safe no-op on non-Windows. Lazy window creation."""

    _registered = False
    _wndproc_ref = None
    _hinstance = None

    def __init__(self):
        self.hwnd = None
        self.visible = False
        self.lines = []
        self._last = None
        self._font = None
        self._shown = False
        self._raised_at = 0.0

    def show(self):
        """Show without activating. Idempotent: no Win32 call when already shown."""
        self.visible = True
        if not _win32():
            return
        self._ensure_window()
        if not self.hwnd or self._shown:
            return
        user32 = _user32()
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        user32.SetWindowPos(
            self.hwnd, _hwnd_topmost(), OSD_X, OSD_Y, OSD_W, OSD_H,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        )
        self._shown = True
        self._raised_at = _now()

    def hide(self):
        self.visible = False
        self._shown = False
        if self.hwnd and _win32():
            _user32().ShowWindow(self.hwnd, SW_HIDE)

    def is_active(self):
        """True only when the Win32 window really exists and is visible on screen."""
        if not self.visible or not self.hwnd or not _win32():
            return False
        try:
            return bool(_user32().IsWindowVisible(self.hwnd))
        except Exception:
            return False

    def update(self, lines):
        """lines: list of (text, color_name) color_name in white|cyan|dim."""
        rows = []
        for item in lines or []:
            if isinstance(item, (tuple, list)) and item:
                rows.append((str(item[0]), str(item[1] if len(item) > 1 else "white")))
            else:
                rows.append((str(item), "white"))
        self.lines = rows
        if rows == self._last:
            self._pump()
            return
        self._last = list(rows)
        if self.hwnd and _win32() and self.visible:
            user32 = _user32()
            # Re-assert z-order at most once per second, only when content changes.
            now = _now()
            if now - self._raised_at >= 1.0:
                user32.SetWindowPos(
                    self.hwnd, _hwnd_topmost(), 0, 0, 0, 0,
                    SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE,
                )
                self._raised_at = now
            user32.InvalidateRect(self.hwnd, None, True)
        self._pump()

    def destroy(self):
        self.visible = False
        self._shown = False
        hwnd = self.hwnd
        self.hwnd = None
        self._last = None
        if hwnd and _win32():
            _instances.pop(int(hwnd), None)
            try:
                _user32().DestroyWindow(hwnd)
            except Exception:
                pass
        if self._font and _win32():
            try:
                _gdi32().DeleteObject(self._font)
            except Exception:
                pass
        self._font = None

    def _ensure_window(self):
        if self.hwnd or not _win32():
            return
        user32 = _user32()
        hinst = self._register_class()
        if hinst is None:
            return
        ex = osd_ex_styles()
        hwnd = user32.CreateWindowExW(
            ex, OSD_CLASS, OSD_TITLE, WS_POPUP,
            OSD_X, OSD_Y, OSD_W, OSD_H,
            None, None, hinst, None,
        )
        if not hwnd:
            return
        self.hwnd = hwnd
        _instances[int(hwnd)] = self
        user32.SetLayeredWindowAttributes(hwnd, COLOR_KEY, 255, LWA_COLORKEY)
        user32.SetWindowPos(
            hwnd, _hwnd_topmost(), OSD_X, OSD_Y, OSD_W, OSD_H, SWP_NOACTIVATE,
        )

    def _register_class(self):
        import ctypes
        from ctypes import wintypes

        user32 = _user32()
        kernel32 = ctypes.windll.kernel32
        kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        hinst = kernel32.GetModuleHandleW(None)
        GameOSD._hinstance = hinst
        if GameOSD._registered:
            return hinst

        WNDPROC = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        )

        def _proc(hwnd, msg, wparam, lparam):
            if msg == WM_PAINT:
                osd = _instances.get(int(hwnd or 0))
                if osd is not None:
                    try:
                        osd._paint(hwnd)
                    except Exception:
                        pass
                    return 0
            if msg == WM_ERASEBKGND:
                return 1
            if msg == WM_NCHITTEST:
                return HTTRANSPARENT
            if msg == WM_CLOSE:
                # Only GameOSD.destroy() may remove the window (S toggles visibility).
                return 0
            if msg == WM_DESTROY:
                osd = _instances.pop(int(hwnd or 0), None)
                if osd is not None and int(osd.hwnd or 0) == int(hwnd or 0):
                    osd.hwnd = None
                    osd._shown = False
                    osd._last = None
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        GameOSD._wndproc_ref = WNDPROC(_proc)

        class WNDCLASSEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.UINT),
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HANDLE),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HANDLE),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
                ("hIconSm", wintypes.HANDLE),
            ]

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = GameOSD._wndproc_ref
        wc.hInstance = hinst
        wc.hbrBackground = _gdi32().GetStockObject(BLACK_BRUSH)
        wc.lpszClassName = OSD_CLASS
        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            err = ctypes.get_last_error()
            if err != ERROR_CLASS_ALREADY_EXISTS:
                return None
        GameOSD._registered = True
        return hinst

    def _paint(self, hwnd):
        import ctypes
        from ctypes import wintypes

        class PAINTSTRUCT(ctypes.Structure):
            _fields_ = [
                ("hdc", wintypes.HDC),
                ("fErase", wintypes.BOOL),
                ("rcPaint", wintypes.RECT),
                ("fRestore", wintypes.BOOL),
                ("fIncUpdate", wintypes.BOOL),
                ("rgbReserved", ctypes.c_byte * 32),
            ]

        user32 = _user32()
        gdi = _gdi32()
        ps = PAINTSTRUCT()
        hdc = user32.BeginPaint(hwnd, ctypes.byref(ps))
        try:
            rect = wintypes.RECT(0, 0, OSD_W, OSD_H)
            brush = gdi.GetStockObject(BLACK_BRUSH)
            user32.FillRect(hdc, ctypes.byref(rect), brush)
            gdi.SetBkMode(hdc, TRANSPARENT_BG)
            if not self._font:
                # NONANTIALIASED_QUALITY: crisp edges with a colour-key window (no black fringe).
                self._font = gdi.CreateFontW(
                    16, 0, 0, 0, 600, 0, 0, 0, 0, 0, 0, NONANTIALIASED_QUALITY, 0, "Segoe UI",
                )
            old = gdi.SelectObject(hdc, self._font)
            flags = DT_LEFT | DT_NOPREFIX | DT_SINGLELINE | DT_END_ELLIPSIS
            y = 8
            for text, tone in self.lines:
                if not text:
                    y += 10
                    continue
                # Thin dark outline keeps the text readable on bright game scenes.
                gdi.SetTextColor(hdc, OUTLINE)
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    shadow = wintypes.RECT(10 + dx, y + dy, OSD_W - 8 + dx, y + 20 + dy)
                    user32.DrawTextW(hdc, text, -1, ctypes.byref(shadow), flags)
                if tone == "cyan":
                    gdi.SetTextColor(hdc, CYAN)
                elif tone == "dim":
                    gdi.SetTextColor(hdc, GRAY)
                else:
                    gdi.SetTextColor(hdc, WHITE)
                box = wintypes.RECT(10, y, OSD_W - 8, y + 20)
                user32.DrawTextW(hdc, text, -1, ctypes.byref(box), flags)
                y += 18
            gdi.SelectObject(hdc, old)
        finally:
            user32.EndPaint(hwnd, ctypes.byref(ps))

    def _pump(self):
        if not _win32() or not self.hwnd:
            return
        import ctypes
        from ctypes import wintypes

        class MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("message", wintypes.UINT),
                ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM),
                ("time", wintypes.DWORD),
                ("pt", wintypes.POINT),
            ]

        user32 = _user32()
        msg = MSG()
        while user32.PeekMessageW(ctypes.byref(msg), self.hwnd, 0, 0, 1):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


def _now():
    from time import monotonic
    return monotonic()


_USER32 = None
_GDI32 = None


def _user32():
    """user32 with 64-bit safe signatures. Configured once."""
    global _USER32
    if _USER32 is not None:
        return _USER32
    import ctypes
    from ctypes import wintypes

    dll = ctypes.WinDLL("user32", use_last_error=True)
    vp = ctypes.c_void_p
    dll.CreateWindowExW.restype = wintypes.HWND
    dll.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    ]
    dll.SetLayeredWindowAttributes.argtypes = [
        wintypes.HWND, wintypes.COLORREF, ctypes.c_ubyte, wintypes.DWORD,
    ]
    dll.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    dll.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    dll.IsWindowVisible.argtypes = [wintypes.HWND]
    dll.DestroyWindow.argtypes = [wintypes.HWND]
    dll.InvalidateRect.argtypes = [wintypes.HWND, vp, wintypes.BOOL]
    dll.BeginPaint.argtypes = [wintypes.HWND, vp]
    dll.BeginPaint.restype = wintypes.HDC
    dll.EndPaint.argtypes = [wintypes.HWND, vp]
    dll.DrawTextW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int, vp, wintypes.UINT]
    dll.PeekMessageW.argtypes = [vp, wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
    dll.TranslateMessage.argtypes = [vp]
    dll.DispatchMessageW.argtypes = [vp]
    dll.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    dll.DefWindowProcW.restype = ctypes.c_ssize_t
    dll.FillRect.argtypes = [wintypes.HDC, vp, wintypes.HBRUSH]
    dll.RegisterClassExW.argtypes = [vp]
    dll.RegisterClassExW.restype = wintypes.ATOM
    _USER32 = dll
    return dll


def _gdi32():
    global _GDI32
    if _GDI32 is not None:
        return _GDI32
    import ctypes
    from ctypes import wintypes

    dll = ctypes.WinDLL("gdi32")
    dll.GetStockObject.argtypes = [ctypes.c_int]
    dll.GetStockObject.restype = wintypes.HGDIOBJ
    dll.CreateFontW.restype = wintypes.HFONT
    dll.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
    dll.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    dll.SelectObject.restype = wintypes.HGDIOBJ
    dll.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    dll.SetBkMode.argtypes = [wintypes.HDC, ctypes.c_int]
    dll.SetTextColor.argtypes = [wintypes.HDC, wintypes.COLORREF]
    _GDI32 = dll
    return dll
