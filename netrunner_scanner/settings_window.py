"""Tk application shell for Net Ready Eyes.

This replaces the old pair of floating Tk helper windows with a single normal
application window.  The video preview is drawn inside the same window that owns
File / Edit / Help menus and the main control panel.
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox, simpledialog
import sys
import queue
import threading
from pathlib import Path
import ctypes
import ctypes.wintypes


import cv2
from PIL import Image, ImageTk

from .camera import list_video_sources
from .runtime_controls import (
    get_settings,
    set_mode,
    set_manual_click_respect_queue,
    set_queue_wait_seconds,
    set_gpu_enabled,
    threshold_definitions,
    get_threshold_values,
    set_threshold_value,
    reset_threshold_values,
    save_settings as save_runtime_settings,
    is_dirty as runtime_settings_dirty,
)
from .gpu_status import configure_gpu
from .roi import (
    rois,
    roi_enabled,
    set_roi_enabled,
    roi_color,
    set_roi_edit_enabled,
    roi_edit_enabled,
    square_up_roi,
    save_rois,
    load_rois,
    default_rois,
    show_roi_labels,
    set_show_roi_labels,
)


def _bgr_to_hex(color):
    b, g, r = color
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _hex_to_bgr(value):
    value = value.strip().lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


class _ConsoleTee:
    def __init__(self, original, callback):
        self.original = original
        self.callback = callback

    def write(self, text):
        try:
            self.original.write(text)
        except Exception:
            pass
        if text and self.callback is not None:
            self.callback(text)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass


class NetReadyEyesWindow:
    def __init__(
        self,
        frame_size_getter,
        mouse_handler,
        load_catalog_callback=None,
        switch_camera_callback=None,
        clear_tracks_callback=None,
        scan_side_callback=None,
        diagnostic_scan_callback=None,
        diagnostic_stop_callback=None,
        save_roi_callback=None,
        load_roi_callback=None,
        reset_roi_callback=None,
        key_callback=None,
        quit_callback=None,
    ):
        self.frame_size_getter = frame_size_getter
        self.mouse_handler = mouse_handler
        self.load_catalog_callback = load_catalog_callback
        self.switch_camera_callback = switch_camera_callback
        self.clear_tracks_callback = clear_tracks_callback
        self.scan_side_callback = scan_side_callback
        self.diagnostic_scan_callback = diagnostic_scan_callback
        self.diagnostic_stop_callback = diagnostic_stop_callback
        self.save_roi_callback = save_roi_callback
        self.load_roi_callback = load_roi_callback
        self.reset_roi_callback = reset_roi_callback
        self.key_callback = key_callback
        self.quit_callback = quit_callback

        self.root = None
        self.video_label = None
        self.photo = None
        self.closed = False
        self.settings_window = None
        self.console_text = None
        self._stdout_original = None
        self._stderr_original = None
        self._console_queue = queue.Queue()
        self._console_owner_thread = None

        self.recognition_mode_var = None
        self.left_click_obs_var = None
        self.queue_wait_var = None
        self.gpu_enabled_var = None

        self.roi_edit_var = None
        self.right_enabled_var = None
        self.show_roi_labels_var = None
        self.status_var = None
        self.threshold_vars = {}
        self.threshold_pixel_previews = {}
        self.threshold_status_var = None
        self._thresholds_canvas = None

        self.diagnostic_mode_var = None
        self._diagnostic_drag_start = None
        self.diagnostic_window = None
        self._diagnostic_images = []
        self._active_diagnostic_region = None
        self._last_live_diagnostic_at = 0.0
        self._live_diagnostic_enabled = True

    def start(self):
        self.root = tk.Tk()
        self._console_owner_thread = threading.get_ident()
        self.root.title("Net Ready Eyes")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Delete>", lambda event: self._send_key("Delete"))
        self.root.bind("<BackSpace>", lambda event: self._send_key("BackSpace"))
        self.root.bind("<Escape>", lambda event: self._send_key("Escape"))
        self.root.bind("<Control-comma>", lambda event: self.open_settings())

        self._build_menu()
        self._build_main_controls()

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)

        self.video_label = ttk.Label(body)
        self.video_label.pack(side="left", fill="both", expand=True)

        self._build_console_panel(body)

        self.video_label.bind("<Motion>", self._on_mouse_motion)
        self.video_label.bind("<ButtonPress-1>", self._on_left_down)
        self.video_label.bind("<ButtonRelease-1>", self._on_left_up)
        self.video_label.bind("<ButtonPress-3>", self._on_right_down)
        self.video_label.focus_set()
        self._install_console_redirect()

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Card Catalog...", command=self.load_catalog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.close)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Video Source...", command=self.open_camera_dialog)
        edit_menu.add_command(label="ROI Settings...", accelerator="Ctrl+,", command=self.open_settings)
        edit_menu.add_command(label="Detection Thresholds...", command=self.open_thresholds_window)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        recognition_menu = tk.Menu(menubar, tearoff=0)
        recognition_menu.add_command(label="Automatic recognition", command=lambda: self.set_recognition_mode("automatic"))
        recognition_menu.add_command(label="Manual recognition", command=lambda: self.set_recognition_mode("manual"))
        recognition_menu.add_separator()
        recognition_menu.add_checkbutton(label="Directed CollectorVision diagnostic mode", command=self.toggle_diagnostic_mode_from_menu)
        menubar.add_cascade(label="Recognition", menu=recognition_menu)

        obs_menu = tk.Menu(menubar, tearoff=0)
        obs_menu.add_command(label="On Left-click: Instant", command=lambda: self.set_left_click_obs_send("Instant"))
        obs_menu.add_command(label="On Left-click: Use queue", command=lambda: self.set_left_click_obs_send("Use queue"))
        obs_menu.add_command(label="Queue wait seconds...", command=self.prompt_queue_wait_seconds)
        obs_menu.add_separator()
        obs_menu.add_command(label="Save runtime settings", command=self.save_runtime_settings)
        menubar.add_cascade(label="OBS", menu=obs_menu)

        performance_menu = tk.Menu(menubar, tearoff=0)
        performance_menu.add_command(label="GPU acceleration on", command=lambda: self.set_gpu_enabled(True))
        performance_menu.add_command(label="GPU acceleration off", command=lambda: self.set_gpu_enabled(False))
        menubar.add_cascade(label="Performance", menu=performance_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_main_controls(self):
        panel = ttk.Frame(self.root, padding=(8, 6))
        panel.pack(fill="x")

        ttk.Button(panel, text="Clear Left Playmat Tracks", command=lambda: self.clear_tracks("left")).pack(side="left", padx=(0, 6))
        ttk.Button(panel, text="Clear Right Playmat Tracks", command=lambda: self.clear_tracks("right")).pack(side="left", padx=(0, 12))

        ttk.Separator(panel, orient="vertical").pack(side="left", fill="y", padx=(0, 10))

        settings = get_settings()
        self.recognition_mode_var = tk.StringVar(value=settings.get("mode", "automatic").title())
        self.left_click_obs_var = tk.StringVar(value="Use queue" if settings.get("manual_click_respect_queue") else "Instant")
        self.queue_wait_var = tk.DoubleVar(value=float(settings.get("queue_wait_seconds", 1.0)))
        self.gpu_enabled_var = tk.BooleanVar(value=bool(settings.get("gpu_enabled", True)))

        ttk.Label(panel, text="Recognition").pack(side="left", padx=(0, 4))
        recognition_combo = ttk.Combobox(panel, textvariable=self.recognition_mode_var, values=["Automatic", "Manual"], state="readonly", width=10)
        recognition_combo.pack(side="left", padx=(0, 10))
        recognition_combo.bind("<<ComboboxSelected>>", lambda _event: self.set_recognition_mode(self.recognition_mode_var.get().lower()))

        ttk.Label(panel, text="On Left-click").pack(side="left", padx=(0, 4))
        obs_combo = ttk.Combobox(panel, textvariable=self.left_click_obs_var, values=["Instant", "Use queue"], state="readonly", width=10)
        obs_combo.pack(side="left", padx=(0, 10))
        obs_combo.bind("<<ComboboxSelected>>", lambda _event: self.set_left_click_obs_send(self.left_click_obs_var.get()))

        ttk.Label(panel, text="Queue wait").pack(side="left", padx=(0, 4))
        wait_spin = ttk.Spinbox(panel, from_=1.0, to=15.0, increment=1.0, textvariable=self.queue_wait_var, width=5, command=self.apply_queue_wait_seconds)
        wait_spin.pack(side="left", padx=(0, 4))
        wait_spin.bind("<Return>", lambda _event: self.apply_queue_wait_seconds())
        wait_spin.bind("<FocusOut>", lambda _event: self.apply_queue_wait_seconds())
        ttk.Label(panel, text="sec").pack(side="left", padx=(0, 10))

        ttk.Checkbutton(panel, text="GPU", variable=self.gpu_enabled_var, command=lambda: self.set_gpu_enabled(self.gpu_enabled_var.get())).pack(side="left", padx=(0, 10))
        ttk.Button(panel, text="Save runtime settings", command=self.save_runtime_settings).pack(side="left", padx=(0, 6))
        ttk.Button(panel, text="ROI Settings", command=self.open_settings).pack(side="left", padx=(0, 6))
        ttk.Button(panel, text="Thresholds", command=self.open_thresholds_window).pack(side="left", padx=(0, 6))
        self.diagnostic_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(panel, text="Directed CV diagnostic", variable=self.diagnostic_mode_var, command=self.announce_diagnostic_mode).pack(side="left", padx=(6, 0))


    def open_thresholds_window(self):
        self.open_settings(select_tab="thresholds")

    def _build_thresholds_tab(self, notebook):
        thresholds_tab = ttk.Frame(notebook)
        notebook.add(thresholds_tab, text="Detection Thresholds")

        canvas = tk.Canvas(thresholds_tab, highlightthickness=0)
        self._thresholds_canvas = canvas
        scrollbar = ttk.Scrollbar(thresholds_tab, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")

        def _configure_scrollregion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _configure_canvas_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        content.bind("<Configure>", _configure_scrollregion)
        canvas.bind("<Configure>", _configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Make the mouse wheel scroll the threshold page instead of doing nothing
        # when the pointer is over labels/spinboxes inside the canvas.
        def _bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", self._on_threshold_mousewheel)
            canvas.bind_all("<Button-4>", self._on_threshold_mousewheel)
            canvas.bind_all("<Button-5>", self._on_threshold_mousewheel)

        def _unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        content.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        values = get_threshold_values()
        self.threshold_vars = {}
        self.threshold_pixel_previews = {}
        grouped = {}
        for definition in threshold_definitions():
            grouped.setdefault(definition.get("group", "Other"), []).append(definition)

        group_order = [
            "Auto acceptance",
            "Boolean guards",
            "Track age-out",
            "Track matching",
            "Card backs / ghosts",
            "Detection candidates",
            "Match quality",
            "Visual recheck",
            "Other",
        ]
        ordered_groups = [g for g in group_order if g in grouped]
        ordered_groups.extend(g for g in grouped.keys() if g not in ordered_groups)

        row = 0
        intro = ttk.Label(
            content,
            text=(
                "These values apply live as you change them. Save runtime settings keeps the current values for the next launch. "
                "The most useful ghost-vs-detection controls are at the top. Pixel thresholds include a small 1:1 preview line so you can see roughly how large the distance is in the camera frame."
            ),
            wraplength=920,
            justify="left",
        )
        intro.grid(row=row, column=0, sticky="ew", padx=12, pady=(12, 8))
        row += 1

        for group in ordered_groups:
            definitions = grouped[group]
            frame = ttk.LabelFrame(content, text=group, padding=(10, 8))
            frame.grid(row=row, column=0, sticky="ew", padx=12, pady=8)
            has_pixel_scale = any(self._threshold_is_pixel_sized(d) for d in definitions)
            frame.columnconfigure(0, minsize=250)
            frame.columnconfigure(1, minsize=110)
            frame.columnconfigure(2, weight=1, minsize=360)
            if has_pixel_scale:
                frame.columnconfigure(3, minsize=190)
            row += 1

            header_font = None
            ttk.Label(frame, text="Option").grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
            ttk.Label(frame, text="Value").grid(row=0, column=1, sticky="w", padx=6, pady=(0, 4))
            ttk.Label(frame, text="What it affects").grid(row=0, column=2, sticky="w", padx=8, pady=(0, 4))
            if has_pixel_scale:
                ttk.Label(frame, text="Pixel scale").grid(row=0, column=3, sticky="w", padx=8, pady=(0, 4))

            inner_row = 1
            for definition in definitions:
                key = definition["key"]
                label = definition.get("label", key)
                kind = definition.get("type", "float")
                current = values.get(key)
                help_text = definition.get("help", "")

                if kind == "bool":
                    var = tk.BooleanVar(value=bool(current))
                    self.threshold_vars[key] = var
                    ttk.Checkbutton(
                        frame,
                        text=label,
                        variable=var,
                        command=lambda k=key: self.apply_threshold_value(k),
                    ).grid(row=inner_row, column=0, sticky="w", padx=4, pady=4)
                    ttk.Label(frame, text="on/off").grid(row=inner_row, column=1, sticky="w", padx=6, pady=4)
                    ttk.Label(frame, text=help_text, wraplength=350, justify="left").grid(row=inner_row, column=2, sticky="nw", padx=8, pady=4)
                    if has_pixel_scale:
                        ttk.Label(frame, text="").grid(row=inner_row, column=3, sticky="w", padx=8, pady=4)
                else:
                    var = tk.DoubleVar(value=float(current)) if kind == "float" else tk.IntVar(value=int(current))
                    self.threshold_vars[key] = var
                    ttk.Label(frame, text=label).grid(row=inner_row, column=0, sticky="w", padx=4, pady=4)

                    spin = ttk.Spinbox(
                        frame,
                        from_=definition.get("min", 0),
                        to=definition.get("max", 9999),
                        increment=definition.get("step", 1),
                        textvariable=var,
                        width=10,
                        command=lambda k=key: self.apply_threshold_value(k),
                    )
                    spin.grid(row=inner_row, column=1, sticky="w", padx=6, pady=4)
                    spin.bind("<Return>", lambda _event, k=key: self.apply_threshold_value(k))
                    spin.bind("<FocusOut>", lambda _event, k=key: self.apply_threshold_value(k))
                    ttk.Label(frame, text=help_text, wraplength=350, justify="left").grid(row=inner_row, column=2, sticky="nw", padx=8, pady=4)

                    if self._threshold_is_pixel_sized(definition):
                        preview = tk.Canvas(frame, width=180, height=22, highlightthickness=0, background="white")
                        preview.grid(row=inner_row, column=3, sticky="w", padx=8, pady=4)
                        self.threshold_pixel_previews[key] = preview
                        self._draw_threshold_pixel_preview(key)
                    elif has_pixel_scale:
                        ttk.Label(frame, text="").grid(row=inner_row, column=3, sticky="w", padx=8, pady=4)

                    # Apply live while typing or using the spinbox arrows. Invalid partial
                    # edits are ignored until the value becomes parseable again.
                    var.trace_add("write", lambda *_args, k=key: self.apply_threshold_value(k, quiet=True))

                inner_row += 1

        button_row = ttk.Frame(content)
        button_row.grid(row=row, column=0, sticky="ew", padx=12, pady=(8, 12))
        ttk.Button(button_row, text="Reset thresholds to defaults", command=self.reset_threshold_values_ui).pack(side="left")
        ttk.Button(button_row, text="Save runtime settings", command=self.save_runtime_settings).pack(side="left", padx=(8, 0))
        self.threshold_status_var = tk.StringVar(value="Changes apply live. Save runtime settings to keep them for next launch.")
        ttk.Label(content, textvariable=self.threshold_status_var, wraplength=920, justify="left").grid(row=row + 1, column=0, sticky="w", padx=12, pady=(0, 12))
        content.columnconfigure(0, weight=1)

    def _threshold_is_pixel_sized(self, definition):
        key = str(definition.get("key", "")).lower()
        label = str(definition.get("label", "")).lower()
        constant = str(definition.get("constant", "")).lower()
        return key.endswith("_px") or "_px" in key or " px" in label or constant.endswith("_px")

    def _on_threshold_mousewheel(self, event):
        canvas = self._thresholds_canvas
        if canvas is None:
            return
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            canvas.yview_scroll(3, "units")
        else:
            delta = getattr(event, "delta", 0)
            if delta:
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    def _draw_threshold_pixel_preview(self, key):
        preview = self.threshold_pixel_previews.get(key)
        var = self.threshold_vars.get(key)
        if preview is None or var is None:
            return
        try:
            value = float(var.get())
        except Exception:
            return
        preview.delete("all")
        width = 180
        height = 22
        left = 6
        usable = width - 42
        shown = max(0, min(usable, int(round(value))))
        y = height // 2
        preview.create_line(left, y, left + shown, y, width=3)
        preview.create_line(left, y - 5, left, y + 5)
        preview.create_line(left + shown, y - 5, left + shown, y + 5)
        label = f"{value:g}px"
        if value > usable:
            label += " +"
        preview.create_text(width - 4, y, text=label, anchor="e")

    def apply_threshold_value(self, key, quiet=False):
        var = self.threshold_vars.get(key)
        if var is None:
            return
        try:
            value = var.get()
        except Exception:
            return
        if set_threshold_value(key, value):
            self._draw_threshold_pixel_preview(key)
            msg = f"Updated threshold: {key} = {get_threshold_values().get(key)}"
            if self.threshold_status_var is not None:
                self.threshold_status_var.set(msg if not quiet else "Changes apply live. Save runtime settings to keep them for next launch.")
            if not quiet:
                print(msg)

    def reset_threshold_values_ui(self):
        if not messagebox.askyesno(
            "Reset thresholds",
            "Reset detection/tracking thresholds to the defaults from config.py?",
            parent=self.settings_window or self.root,
        ):
            return
        reset_threshold_values()
        values = get_threshold_values()
        for key, var in self.threshold_vars.items():
            if key not in values:
                continue
            try:
                var.set(values[key])
            except Exception:
                pass
            self._draw_threshold_pixel_preview(key)
        if self.threshold_status_var is not None:
            self.threshold_status_var.set("Reset thresholds to defaults. Click Save runtime settings to keep this for next launch.")
        print("Reset runtime thresholds to defaults.")

    def _build_console_panel(self, parent):
        panel = ttk.Frame(parent, padding=(8, 0, 8, 8), width=360)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        ttk.Label(panel, text="Console Output").pack(anchor="w", pady=(0, 4))
        frame = ttk.Frame(panel)
        frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(frame, orient="vertical", width=18)
        self.console_text = tk.Text(frame, width=44, height=18, wrap="word", yscrollcommand=scrollbar.set, state="disabled")
        scrollbar.config(command=self.console_text.yview)
        self.console_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _install_console_redirect(self):
        if self._stdout_original is not None:
            return
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        sys.stdout = _ConsoleTee(sys.stdout, self.append_console)
        sys.stderr = _ConsoleTee(sys.stderr, self.append_console)

    def append_console(self, text):
        if not text or self.closed:
            return
        # Worker threads can print while recognition is running.  Tk widgets may
        # only be touched from the thread that owns the Tk root, so queue all
        # console writes and flush them during the normal UI pump.
        try:
            self._console_queue.put_nowait(str(text))
        except Exception:
            pass

    def _flush_console_queue(self):
        if self.console_text is None or self.closed:
            return
        parts = []
        try:
            while len(parts) < 200:
                parts.append(self._console_queue.get_nowait())
        except queue.Empty:
            pass
        except Exception:
            return

        if not parts:
            return
        try:
            self.console_text.configure(state="normal")
            self.console_text.insert("end", "".join(parts))
            self.console_text.see("end")
            self.console_text.configure(state="disabled")
        except tk.TclError:
            pass

    def _cursor_xy(self):
        """Return current pointer position in virtual-screen coordinates."""
        try:
            point = ctypes.wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
                return int(point.x), int(point.y)
        except Exception:
            pass
        try:
            return int(self.root.winfo_pointerx()), int(self.root.winfo_pointery())
        except Exception:
            return 120, 120

    def _position_near_mouse(self, win, width=None, height=None):
        # Position child dialogs where the user clicked.  Tk's normal screen
        # clamping can be unreliable on mixed-DPI / multi-monitor setups, so
        # this uses the OS cursor position first and only nudges the window if
        # it would be obviously off the visible virtual desktop.
        try:
            self.root.update_idletasks()
        except Exception:
            pass

        x, y = self._cursor_xy()
        x += 16
        y += 16

        if width is not None and height is not None:
            width = int(width)
            height = int(height)
            try:
                user32 = ctypes.windll.user32
                left = int(user32.GetSystemMetrics(76))   # SM_XVIRTUALSCREEN
                top = int(user32.GetSystemMetrics(77))    # SM_YVIRTUALSCREEN
                vwidth = int(user32.GetSystemMetrics(78)) # SM_CXVIRTUALSCREEN
                vheight = int(user32.GetSystemMetrics(79))# SM_CYVIRTUALSCREEN
                right = left + vwidth
                bottom = top + vheight
                if x + width > right - 24:
                    x = max(left + 24, right - width - 24)
                if y + height > bottom - 48:
                    y = max(top + 24, bottom - height - 48)
                x = max(left + 8, x)
                y = max(top + 8, y)
            except Exception:
                pass
            win.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
        else:
            win.geometry(f"+{int(x)}+{int(y)}")

    def refresh_runtime_vars(self):
        settings = get_settings()
        if self.recognition_mode_var is not None:
            self.recognition_mode_var.set(settings.get("mode", "automatic").title())
        if self.left_click_obs_var is not None:
            self.left_click_obs_var.set("Use queue" if settings.get("manual_click_respect_queue") else "Instant")
        if self.queue_wait_var is not None:
            self.queue_wait_var.set(float(settings.get("queue_wait_seconds", 1.0)))
        if self.gpu_enabled_var is not None:
            self.gpu_enabled_var.set(bool(settings.get("gpu_enabled", True)))

    def set_recognition_mode(self, mode):
        mode = str(mode).lower()
        if set_mode(mode):
            self.refresh_runtime_vars()
            print(f"Recognition mode set to {mode}.")

    def set_left_click_obs_send(self, value):
        use_queue = str(value).lower().startswith("use")
        set_manual_click_respect_queue(use_queue)
        self.refresh_runtime_vars()
        print("On Left-click uses queue." if use_queue else "On Left-click sends instantly.")

    def apply_queue_wait_seconds(self):
        try:
            value = float(self.queue_wait_var.get())
        except Exception:
            value = get_settings().get("queue_wait_seconds", 1.0)
        set_queue_wait_seconds(value)
        self.refresh_runtime_vars()
        print(f"Queue wait seconds set to {float(get_settings().get('queue_wait_seconds', value)):.1f}.")

    def prompt_queue_wait_seconds(self):
        value = simpledialog.askfloat(
            "Queue wait seconds",
            "Seconds to wait between OBS queue sends:",
            initialvalue=float(get_settings().get("queue_wait_seconds", 1.0)),
            minvalue=1.0,
            maxvalue=15.0,
            parent=self.root,
        )
        if value is not None:
            set_queue_wait_seconds(value)
            self.refresh_runtime_vars()
            print(f"Queue wait seconds set to {float(get_settings().get('queue_wait_seconds', value)):.1f}.")

    def set_gpu_enabled(self, enabled):
        set_gpu_enabled(bool(enabled))
        configure_gpu(bool(enabled))
        self.refresh_runtime_vars()
        print("GPU acceleration enabled." if enabled else "GPU acceleration disabled.")

    def save_runtime_settings(self):
        save_runtime_settings()
        self.refresh_runtime_vars()

    def diagnostic_mode_enabled(self):
        try:
            return bool(self.diagnostic_mode_var is not None and self.diagnostic_mode_var.get())
        except Exception:
            return False

    def announce_diagnostic_mode(self):
        if self.diagnostic_mode_enabled():
            print("Directed CollectorVision diagnostic mode on. Left-drag a tight box around a problem card.")
        else:
            print("Directed CollectorVision diagnostic mode off.")

    def toggle_diagnostic_mode_from_menu(self):
        if self.diagnostic_mode_var is None:
            self.diagnostic_mode_var = tk.BooleanVar(value=False)
        self.diagnostic_mode_var.set(not self.diagnostic_mode_var.get())
        self.announce_diagnostic_mode()

    def _dispatch_diagnostic_mouse(self, cv_event, event):
        if cv_event == cv2.EVENT_LBUTTONDOWN:
            self._diagnostic_drag_start = (int(event.x), int(event.y))
            return True
        if cv_event == cv2.EVENT_MOUSEMOVE:
            return self._diagnostic_drag_start is not None
        if cv_event == cv2.EVENT_LBUTTONUP and self._diagnostic_drag_start is not None:
            x1, y1 = self._diagnostic_drag_start
            x2, y2 = int(event.x), int(event.y)
            self._diagnostic_drag_start = None
            self._active_diagnostic_region = (x1, y1, x2, y2)
            self._live_diagnostic_enabled = True
            self._last_live_diagnostic_at = 0.0
            if self.diagnostic_scan_callback is not None:
                self.diagnostic_scan_callback(x1, y1, x2, y2)
            return True
        return False


    def _open_direct_diagnostic_window(self):
        if self.root is None:
            return None
        if self.diagnostic_window is not None:
            try:
                if self.diagnostic_window.winfo_exists():
                    self.diagnostic_window.lift()
                    return self.diagnostic_window
            except tk.TclError:
                pass

        win = tk.Toplevel(self.root)
        win.title("Directed CollectorVision Diagnostics")
        win.geometry("1160x780")
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_direct_diagnostic_window())

        header = ttk.Frame(win, padding=(10, 8))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Directed CollectorVision Diagnostics",
            font=("Segoe UI", 12, "bold"),
        ).pack(side="left")
        ttk.Button(header, text="Clear", command=self.clear_direct_diagnostic_display).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Stop live", command=self.stop_live_direct_diagnostic).pack(side="right", padx=(6, 0))
        ttk.Button(header, text="Close", command=self._close_direct_diagnostic_window).pack(side="right")

        body = ttk.Frame(win, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        ttk.Label(left, text="Full frame with CollectorVision box").pack(anchor="w")
        win.full_frame_label = ttk.Label(left, anchor="center")
        win.full_frame_label.pack(fill="both", expand=True)

        right = ttk.Frame(body, width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        grid = ttk.Frame(right)
        grid.pack(fill="x")
        for row, title in enumerate(("Your DIRECT CV crop", "CollectorVision dewarped crop", "All cardinal rotations", "Top matched catalog image")):
            ttk.Label(grid, text=title).grid(row=row * 2, column=0, sticky="w", pady=(0 if row == 0 else 8, 2))
            label = ttk.Label(grid, anchor="center")
            label.grid(row=row * 2 + 1, column=0, sticky="ew")
            setattr(win, ["crop_label", "dewarp_label", "rotations_label", "match_label"][row], label)

        ttk.Label(right, text="Recognition details").pack(anchor="w", pady=(10, 2))
        text_frame = ttk.Frame(right)
        text_frame.pack(fill="both", expand=True)
        scrollbar = tk.Scrollbar(text_frame, orient="vertical")
        details = tk.Text(text_frame, width=52, height=16, wrap="word", yscrollcommand=scrollbar.set)
        scrollbar.config(command=details.yview)
        details.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        win.details_text = details

        self.diagnostic_window = win
        return win

    def _close_direct_diagnostic_window(self):
        if self.diagnostic_window is not None:
            try:
                self.diagnostic_window.destroy()
            except tk.TclError:
                pass
        self.diagnostic_window = None
        self._diagnostic_images = []
        self._active_diagnostic_region = None
        self._last_live_diagnostic_at = 0.0
        self._live_diagnostic_enabled = True

    def _photo_from_pil(self, pil_image, max_size):
        if pil_image is None:
            return None
        try:
            image = pil_image.copy()
            image.thumbnail(max_size, Image.LANCZOS)
            return ImageTk.PhotoImage(image=image)
        except Exception:
            return None

    def _set_image_label_from_preview(self, label, pil_image, max_size, empty_text="not available"):
        photo = self._photo_from_pil(pil_image, max_size)
        if photo is None:
            label.configure(image="", text=empty_text)
            label.image = None
            return False
        label.configure(image=photo, text="")
        label.image = photo
        self._diagnostic_images.append(photo)
        return True

    def _load_preview_photo(self, path, max_size):
        if not path:
            return None
        try:
            image = Image.open(Path(path))
            image.thumbnail(max_size, Image.LANCZOS)
            return ImageTk.PhotoImage(image=image)
        except Exception:
            return None

    def _set_image_label(self, label, path, max_size, empty_text="not available"):
        photo = self._load_preview_photo(path, max_size)
        if photo is None:
            label.configure(image="", text=empty_text)
            label.image = None
            return
        label.configure(image=photo, text="")
        label.image = photo
        self._diagnostic_images.append(photo)

    def show_direct_diagnostic_report(self, report):
        win = self._open_direct_diagnostic_window()
        if win is None:
            return
        self._diagnostic_images = []
        previews = report.get("_preview_images") or {}
        if not self._set_image_label_from_preview(win.full_frame_label, previews.get("overlay"), (720, 640), empty_text="no full-frame overlay"):
            self._set_image_label(win.full_frame_label, report.get("overlay"), (720, 640), empty_text="no full-frame overlay")
        if not self._set_image_label_from_preview(win.crop_label, previews.get("crop_overlay") or previews.get("source_crop"), (390, 210), empty_text="no crop"):
            self._set_image_label(win.crop_label, report.get("crop_overlay") or report.get("source_crop"), (390, 210), empty_text="no crop")
        if not self._set_image_label_from_preview(win.dewarp_label, previews.get("dewarped"), (260, 190), empty_text="CollectorVision did not return a dewarped card"):
            self._set_image_label(win.dewarp_label, report.get("dewarped"), (260, 190), empty_text="CollectorVision did not return a dewarped card")
        if not self._set_image_label_from_preview(win.rotations_label, previews.get("rotations"), (390, 160), empty_text="no rotation preview"):
            self._set_image_label(win.rotations_label, report.get("rotations"), (390, 160), empty_text="no rotation preview")
        self._set_image_label(win.match_label, report.get("matched_card"), (260, 190), empty_text="matched catalog image not found on disk")

        summary = report.get("summary") or {}
        margin = summary.get("margin")
        margin_text = "n/a" if margin is None else f"{float(margin):.3f}"
        lines = [
            f"Region: {report.get('region')}",
            f"CollectorVision reason: {report.get('collectorvision_reason')}",
            f"Result: {summary.get('id')}",
            f"Score: {float(summary.get('score') or 0.0):.3f}",
            f"Margin: {margin_text}",
            f"Best tested rotation: {summary.get('rotation')}",
            f"Sharpness: {summary.get('sharpness')}",
            f"Confidence: {summary.get('confidence')}",
            "",
            "Top alternatives:",
        ]
        for alt in (summary.get("alternatives") or [])[:12]:
            lines.append(f"  {alt.get('id')}   score={float(alt.get('score') or 0.0):.3f}   rot={alt.get('rotation')}")
        lines.extend([
            "",
            "Saved files:" if not report.get("live") else "Live preview: not saving frames until Stop live.",
            f"  {report.get('source_crop')}" if not report.get("live") else "",
            f"  {report.get('crop_overlay')}" if not report.get("live") else "",
            f"  {report.get('dewarped')}" if not report.get("live") else "",
            f"  {report.get('overlay')}" if not report.get("live") else "",
        ])
        win.details_text.configure(state="normal")
        win.details_text.delete("1.0", "end")
        win.details_text.insert("end", "\n".join(lines))
        win.details_text.configure(state="disabled")
        if not report.get("live"):
            try:
                win.lift()
            except tk.TclError:
                pass


    def stop_live_direct_diagnostic(self):
        region = self._active_diagnostic_region
        if region is not None and self.diagnostic_stop_callback is not None:
            try:
                self.diagnostic_stop_callback(region)
            except Exception as exc:
                print(f"Failed to save final directed diagnostic frame: {exc}")
        self._live_diagnostic_enabled = False
        self._active_diagnostic_region = None
        print("Directed CollectorVision live diagnostic stopped. Final frame saved if a live region was active.")

    def get_active_direct_diagnostic_region(self):
        if not self._live_diagnostic_enabled:
            return None
        if self.diagnostic_window is None:
            return None
        return self._active_diagnostic_region

    def live_direct_diagnostic_due(self, now_seconds, interval_seconds=0.45):
        if self.get_active_direct_diagnostic_region() is None:
            return False
        if now_seconds - self._last_live_diagnostic_at < interval_seconds:
            return False
        self._last_live_diagnostic_at = now_seconds
        return True

    def clear_direct_diagnostic_display(self):
        if self.diagnostic_window is None:
            return
        win = self.diagnostic_window
        for attr in ("full_frame_label", "crop_label", "dewarp_label", "rotations_label", "match_label"):
            label = getattr(win, attr, None)
            if label is not None:
                label.configure(image="", text="")
                label.image = None
        self._diagnostic_images = []
        try:
            win.details_text.configure(state="normal")
            win.details_text.delete("1.0", "end")
            win.details_text.configure(state="disabled")
        except tk.TclError:
            pass

    def _send_key(self, value):
        if self.key_callback is not None:
            self.key_callback(value)
        return "break"

    def _on_key(self, event):
        value = event.keysym if event.keysym in ("Delete", "BackSpace", "Escape") else event.char
        if not value:
            value = event.keysym
        return self._send_key(value)

    def _mouse_flags(self, event):
        flags = 0
        # Tk's state bitmask: Shift=0x0001, Ctrl=0x0004, Alt often 0x0008.
        if event.state & 0x0001:
            flags |= cv2.EVENT_FLAG_SHIFTKEY
        if event.state & 0x0004:
            flags |= cv2.EVENT_FLAG_CTRLKEY
        if event.state & 0x0008:
            flags |= cv2.EVENT_FLAG_ALTKEY
        return flags

    def _dispatch_mouse(self, cv_event, event):
        if self.diagnostic_mode_enabled():
            consumed = self._dispatch_diagnostic_mouse(cv_event, event)
            self.video_label.focus_set()
            if consumed:
                return
        if self.mouse_handler is not None:
            self.mouse_handler(cv_event, int(event.x), int(event.y), self._mouse_flags(event), None)
        self.video_label.focus_set()

    def _on_mouse_motion(self, event):
        self._dispatch_mouse(cv2.EVENT_MOUSEMOVE, event)

    def _on_left_down(self, event):
        self._dispatch_mouse(cv2.EVENT_LBUTTONDOWN, event)

    def _on_left_up(self, event):
        self._dispatch_mouse(cv2.EVENT_LBUTTONUP, event)

    def _on_right_down(self, event):
        self._dispatch_mouse(cv2.EVENT_RBUTTONDOWN, event)

    def update_frame(self, bgr_frame):
        if self.closed or self.root is None:
            return False
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.photo)
        self.video_label.image = self.photo
        self.pump()
        return not self.closed

    def pump(self):
        if self.closed or self.root is None:
            return
        try:
            self._flush_console_queue()
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True

    def close(self):
        try:
            self._flush_console_queue()
        except Exception:
            pass
        self.closed = True
        if self._stdout_original is not None:
            sys.stdout = self._stdout_original
            self._stdout_original = None
        if self._stderr_original is not None:
            sys.stderr = self._stderr_original
            self._stderr_original = None
        if self.quit_callback is not None:
            self.quit_callback()
        try:
            if self.root is not None:
                self.root.destroy()
        except tk.TclError:
            pass

    def load_catalog(self):
        path = filedialog.askopenfilename(
            title="Load card catalog",
            filetypes=[("CollectorVision / Net Ready Eyes catalog", "*.npz"), ("All files", "*.*")],
        )
        if path and self.load_catalog_callback is not None:
            self.load_catalog_callback(path)

    def open_camera_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Video Source")
        win.transient(self.root)
        self._position_near_mouse(win, width=860, height=220)
        win.protocol("WM_DELETE_WINDOW", win.destroy)
        win.grab_set()

        ttk.Label(win, text="Choose video source:").pack(anchor="w", padx=12, pady=(12, 6))

        try:
            sources = list_video_sources(probe=False)
        except Exception as exc:
            sources = []
            print(f"Could not enumerate video sources: {exc}")

        display = [source["label"] for source in sources] or ["No video sources found"]
        selected = tk.StringVar(value=display[0])
        combo = ttk.Combobox(win, textvariable=selected, values=display, state="readonly", width=112)
        combo.pack(fill="x", padx=12, pady=6)

        def apply():
            if not sources:
                win.destroy()
                return
            try:
                idx = int(selected.get().split(":", 1)[0])
            except Exception:
                idx = int(sources[0].get("index", 0))
            if self.switch_camera_callback is not None:
                self.switch_camera_callback(idx)
            win.destroy()

        buttons = ttk.Frame(win)
        buttons.pack(fill="x", padx=12, pady=12)
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(buttons, text="Use Video Source", command=apply).pack(side="right")

    def show_about(self):
        messagebox.showinfo(
            "About Net Ready Eyes",
            "Net Ready Eyes\nCard identification tool for livestreaming card games.",
            parent=self.root,
        )

    def scan_side(self, side):
        if self.scan_side_callback is not None:
            self.scan_side_callback(side)

    def clear_tracks(self, side):
        if self.clear_tracks_callback is not None:
            count = self.clear_tracks_callback(side)
            print(f"Cleared {side} playmat tracks ({count}).")

    def save_roi(self):
        if self.save_roi_callback is not None:
            self.save_roi_callback()
        else:
            save_rois()

    def open_settings(self, select_tab=None):
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.refresh_vars()
            self.settings_window.lift()
            return

        win = tk.Toplevel(self.root)
        win.title("Settings")
        win.transient(self.root)
        win.minsize(1180, 720)
        self._position_near_mouse(win, width=1280, height=820)
        self.settings_window = win

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        roi_tab = ttk.Frame(notebook)
        notebook.add(roi_tab, text="ROI")
        self._build_thresholds_tab(notebook)
        if select_tab == "thresholds":
            notebook.select(1)

        self.roi_edit_var = tk.BooleanVar(value=roi_edit_enabled())
        self.right_enabled_var = tk.BooleanVar(value=roi_enabled(rois.get("right")))
        self.show_roi_labels_var = tk.BooleanVar(value=show_roi_labels())
        self.status_var = tk.StringVar(value="")

        ttk.Checkbutton(
            roi_tab,
            text="Enable ROI edit mode (move / dewarp corners)",
            variable=self.roi_edit_var,
            command=self.apply_roi_edit_toggle,
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(12, 6))

        ttk.Checkbutton(
            roi_tab,
            text="Enable right / second ROI",
            variable=self.right_enabled_var,
            command=self.apply_right_enabled,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=6)

        ttk.Checkbutton(
            roi_tab,
            text='Show "LEFT" / "RIGHT" ROI labels',
            variable=self.show_roi_labels_var,
            command=self.apply_show_roi_labels,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=12, pady=6)

        ttk.Separator(roi_tab).grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=8)

        ttk.Label(roi_tab, text="ROI color:").grid(row=4, column=0, sticky="w", padx=12, pady=6)
        ttk.Button(roi_tab, text="Left color...", command=lambda: self.pick_roi_color("left")).grid(row=4, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(roi_tab, text="Right color...", command=lambda: self.pick_roi_color("right")).grid(row=4, column=2, sticky="ew", padx=6, pady=6)

        ttk.Label(roi_tab, text="Square/reset:").grid(row=5, column=0, sticky="w", padx=12, pady=6)
        ttk.Button(roi_tab, text="Square left", command=lambda: self.square_roi("left")).grid(row=5, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(roi_tab, text="Square right", command=lambda: self.square_roi("right")).grid(row=5, column=2, sticky="ew", padx=6, pady=6)

        ttk.Button(roi_tab, text="Reset both ROIs to defaults", command=self.reset_rois).grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=6)
        ttk.Button(roi_tab, text="Save ROI settings", command=self.save).grid(row=7, column=0, sticky="ew", padx=12, pady=6)
        ttk.Button(roi_tab, text="Load ROI settings", command=self.load).grid(row=7, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(roi_tab, text="Close", command=win.destroy).grid(row=7, column=2, sticky="ew", padx=6, pady=6)

        ttk.Label(
            roi_tab,
            text="Normal drag selects card tracks. Turn ROI edit mode on only while positioning playmat ROIs, then turn it off before normal operation.",
            wraplength=460,
        ).grid(row=8, column=0, columnspan=3, sticky="w", padx=12, pady=(10, 4))
        ttk.Label(roi_tab, textvariable=self.status_var, wraplength=460).grid(row=9, column=0, columnspan=3, sticky="w", padx=12, pady=(2, 8))

        for col in range(3):
            roi_tab.columnconfigure(col, weight=1)

        self.refresh_vars()

    def refresh_vars(self):
        if self.roi_edit_var is not None:
            self.roi_edit_var.set(roi_edit_enabled())
        if self.right_enabled_var is not None:
            self.right_enabled_var.set(roi_enabled(rois.get("right")))
        if self.show_roi_labels_var is not None:
            self.show_roi_labels_var.set(show_roi_labels())

    def set_status(self, text):
        if self.status_var is not None:
            self.status_var.set(text)
        if text:
            print(text)

    def apply_show_roi_labels(self):
        enabled = set_show_roi_labels(self.show_roi_labels_var.get())
        self.set_status('ROI labels are visible.' if enabled else 'ROI labels are hidden.')

    def apply_roi_edit_toggle(self):
        enabled = set_roi_edit_enabled(self.roi_edit_var.get())
        self.set_status("ROI edit mode on." if enabled else "ROI edit mode off.")

    def apply_right_enabled(self):
        set_roi_enabled("right", self.right_enabled_var.get())
        self.set_status("Right ROI enabled." if self.right_enabled_var.get() else "Right ROI disabled.")

    def pick_roi_color(self, side):
        roi = rois.get(side)
        if roi is None:
            return
        initial = _bgr_to_hex(roi_color(roi))
        _rgb, hex_value = colorchooser.askcolor(color=initial, title=f"Choose {side} ROI color", parent=self.settings_window)
        if not hex_value:
            return
        bgr = _hex_to_bgr(hex_value)
        roi["color"] = [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        self.set_status(f"Updated {side} ROI color.")

    def square_roi(self, side):
        square_up_roi(side)
        self.set_status(f"Squared {side} ROI to its bounding rectangle.")

    def reset_rois(self):
        frame_width, frame_height = self.frame_size_getter()
        rois.update(default_rois(frame_width, frame_height))
        self.refresh_vars()
        self.set_status("Reset both ROIs to default left/right split.")

    def save(self):
        save_rois()
        self.set_status("Saved ROI settings.")

    def load(self):
        frame_width, frame_height = self.frame_size_getter()
        loaded = load_rois(frame_width, frame_height)
        rois["left"] = loaded["left"]
        rois["right"] = loaded["right"]
        self.refresh_vars()
        self.set_status("Loaded ROI settings.")


# Backward-compatible name for older imports.
SettingsController = NetReadyEyesWindow
