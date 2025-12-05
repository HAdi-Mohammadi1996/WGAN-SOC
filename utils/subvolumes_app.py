import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import functions from the separate logic file
import image_ops

# --- UI CONSTANTS ---
MONO_FONT = ("Menlo", 11)
TITLE_FONT = ("Menlo", 13, "bold")
THEME_BG = "#0a0e1a"
PANEL_BG = "#141b27"
CANVAS_BG = "#000000"
TEXT_COLOR = "#e0e0e0"
SUBTEXT_COLOR = "#64748b"
ACCENT_COLOR = "#00ff88"
ROI_COLOR_SELECT = "#00ff88"
ROI_COLOR_NORMAL = "#64748b"
GRID_SELECT_COLOR = "#ff006e"
GRID_OUTLINE_COLOR = "#64748b"
HANDLE_FILL = "#0a0e1a"
HANDLE_BORDER = "#00ff88"
HANDLE_SIZE = 8
MIN_ROI_SIZE = 10
MIN_GRID_SIZE = 20


class SubvolumeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Subvolume Extractor")
        self.root.configure(bg=THEME_BG)
        self.root.geometry("1600x900")
        self._maximize_window()

        # --- Styles ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Main.TFrame", background=PANEL_BG)
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=TEXT_COLOR, font=MONO_FONT)
        style.configure("Heading.TLabel", font=TITLE_FONT, foreground=ACCENT_COLOR, background=PANEL_BG)
        style.configure("TButton", background="#1b2639", foreground=TEXT_COLOR, font=MONO_FONT, padding=6)
        style.map("TButton", background=[("active", "#223149")])
        style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="#041925", font=TITLE_FONT, padding=10)
        style.map("Accent.TButton", background=[("active", "#1dffd4")])
        style.configure("TEntry", fieldbackground="#0e1625", foreground=TEXT_COLOR, insertcolor=ACCENT_COLOR)
        style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_COLOR, font=MONO_FONT)
        style.configure("Horizontal.TProgressbar", background=ACCENT_COLOR, troughcolor="#0e1625", bordercolor=PANEL_BG)

        # --- Data State ---
        self.image_paths = []
        self.cv_image = None
        self.current_display_img = None
        self.current_image_index = 0
        
        self.aligned_images = []        
        self.aligned_images_basic = [] 
        self.aligned_images_final = [] 
        
        self.raw_angles = []
        self.final_angles = []
        self.angle_outliers = []
        self.corrected_frames = 0
        self.alignment_ready = False
        self.reference_index = 0
        self.folder_path = ""
        
        # --- ROI State ---
        self.rois = []
        self.grid_groups = []
        self.scale_factor = 1.0
        self.drag_mode = None
        self.selected_roi_idx = None
        self.selected_group = None
        self.start_x = 0
        self.start_y = 0
        self.temp_rect_id = None
        self.active_handle = None
        self.resize_origin = None

        # --- Layout Architecture ---
        
        # 1. Left Control Panel (Scrollable)
        self.left_container = tk.Frame(root, bg=PANEL_BG, width=320)
        self.left_container.pack(side=tk.LEFT, fill=tk.Y)
        self.left_container.pack_propagate(False) # Force width

        self._create_scrollable_control_panel()

        # 2. Right Visualization Panel
        self.viz_panel = ttk.Frame(root, style="Main.TFrame", width=450, padding=16)
        self.viz_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # 3. Center Canvas (Responsive)
        self.canvas_panel = tk.Frame(root, bg=CANVAS_BG)
        self.canvas_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_panel, bg=CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event to handle dynamic scaling
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Mouse Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Keyboard Bindings
        self.root.bind("<Left>", lambda e: self.navigate_image(-1))
        self.root.bind("<Right>", lambda e: self.navigate_image(1))
        self.root.bind("<Up>", lambda e: self.navigate_image(-10))
        self.root.bind("<Down>", lambda e: self.navigate_image(10))
        
        # Mouse Wheel for Scrollbar (Linux/Windows/Mac handling)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)

        self._build_ui_content()
        self._build_viz_panel()

    def _create_scrollable_control_panel(self):
        """Creates the scrollable infrastructure for the left panel."""
        self.control_canvas = tk.Canvas(self.left_container, bg=PANEL_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_container, orient="vertical", command=self.control_canvas.yview)
        self.control_panel = ttk.Frame(self.control_canvas, style="Main.TFrame", padding=16)

        self.control_canvas.create_window((0, 0), window=self.control_panel, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Update scrollregion when content changes
        self.control_panel.bind("<Configure>", 
            lambda e: self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all")))
        
        # Ensure inner frame matches canvas width
        self.control_canvas.bind("<Configure>", 
            lambda e: self.control_canvas.itemconfig(self.control_canvas.find_all()[0], width=e.width))

    def _on_mousewheel(self, event):
        """Global mousewheel handler."""
        # Only scroll if mouse is over the left panel
        x, y = self.root.winfo_pointerxy()
        widget = self.root.winfo_containing(x, y)
        if str(widget).startswith(str(self.left_container)):
            if event.num == 5 or event.delta < 0:
                self.control_canvas.yview_scroll(1, "units")
            elif event.num == 4 or event.delta > 0:
                self.control_canvas.yview_scroll(-1, "units")

    def _build_ui_content(self):
        """Builds the widgets inside the scrollable control panel."""
        # Variables
        self.method_var = tk.StringVar(value="temporal_smooth")
        self.window_var = tk.StringVar(value="5")
        self.deviation_var = tk.StringVar(value="1.0")
        self.jump_var = tk.StringVar(value="2.0")
        self.view_mode_var = tk.StringVar(value="Original")
        self.vert_window_var = tk.StringVar(value="5")
        
        # Header
        ttk.Label(self.control_panel, text="Alignment + ROI Workflow", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
        
        # [0] Alignment Parameters
        ttk.Label(self.control_panel, text="[0] Alignment Parameters", font=TITLE_FONT).pack(anchor="w", pady=(4, 2))
        frm_params = ttk.Frame(self.control_panel, style="Main.TFrame")
        frm_params.pack(fill=tk.X, pady=4)
        
        ttk.Label(frm_params, text="Method").grid(row=0, column=0, sticky="w")
        self.cmb_method = ttk.Combobox(frm_params, textvariable=self.method_var, 
                                       values=("temporal_smooth", "moving_average", "phase_correlation"), 
                                       state="readonly", width=18)
        self.cmb_method.grid(row=0, column=1, padx=6, pady=2, sticky="ew")
        
        ttk.Label(frm_params, text="Window").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm_params, textvariable=self.window_var, width=6).grid(row=1, column=1, sticky="w", padx=6)
        
        ttk.Label(frm_params, text="Max Dev").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm_params, textvariable=self.deviation_var, width=6).grid(row=2, column=1, sticky="w", padx=6)
        
        ttk.Label(frm_params, text="Max Jump").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm_params, textvariable=self.jump_var, width=6).grid(row=3, column=1, sticky="w", padx=6)
        
        frm_params.grid_columnconfigure(1, weight=1)
        
        # [1] Select & Align
        ttk.Label(self.control_panel, text="[1] Select & Align", font=TITLE_FONT).pack(anchor="w", pady=(8, 2))
        frm_align = ttk.Frame(self.control_panel, style="Main.TFrame")
        frm_align.pack(fill=tk.X, pady=4)
        
        ttk.Button(frm_align, text="Select Folder", command=self.load_folder).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        self.btn_align = ttk.Button(frm_align, text="Align Volume", command=self.align_sequence, state=tk.DISABLED)
        self.btn_align.grid(row=0, column=1, sticky="ew")
        frm_align.grid_columnconfigure(0, weight=1)
        frm_align.grid_columnconfigure(1, weight=1)

        # Extra Alignment Tools
        ttk.Label(self.control_panel, text="Vertical Median Window:").pack(anchor="w", pady=(8, 2))
        ttk.Entry(self.control_panel, textvariable=self.vert_window_var).pack(fill=tk.X, pady=(0, 4))
        
        self.btn_align_tops = ttk.Button(self.control_panel, text="Align Tops (Fix Jitter)", 
                                         command=self.align_vertical_jitter, state=tk.DISABLED)
        self.btn_align_tops.pack(fill=tk.X, pady=(2, 6))
        
        # View Toggle
        ttk.Label(self.control_panel, text="View Mode:").pack(anchor="w", pady=(6, 2))
        self.cmb_view = ttk.Combobox(self.control_panel, textvariable=self.view_mode_var, 
                                     values=("Original", "Aligned (Rotation)", "Aligned (Full)"),
                                     state="readonly")
        self.cmb_view.pack(fill=tk.X, pady=(0, 6))
        self.cmb_view.bind("<<ComboboxSelected>>", lambda e: self.toggle_view_mode())
        
        self.btn_plot = ttk.Button(self.control_panel, text="Show Angle Plot", command=self.show_angle_plot, state=tk.DISABLED)
        self.btn_plot.pack(fill=tk.X, pady=(2, 6))
        
        self.lbl_status = ttk.Label(self.control_panel, text="Step 1: select folder.", 
                                    foreground=SUBTEXT_COLOR, wraplength=280)
        self.lbl_status.pack(anchor="w", pady=(0, 12))
        
        ttk.Separator(self.control_panel).pack(fill="x", pady=8)
        
        # [2] ROI Tools
        ttk.Label(self.control_panel, text="[2] ROI Tools", font=TITLE_FONT).pack(anchor="w", pady=(4, 6))
        frm_coords = ttk.Frame(self.control_panel, style="Main.TFrame")
        frm_coords.pack(fill=tk.X, pady=4)
        
        self.entries = {}
        for i, label in enumerate(["X", "Y", "W", "H"]):
            ttk.Label(frm_coords, text=label).grid(row=0, column=i * 2, padx=(0, 2))
            entry = ttk.Entry(frm_coords, width=5)
            entry.grid(row=0, column=i * 2 + 1, padx=(0, 6))
            self.entries[label] = entry

        ttk.Button(self.control_panel, text="Add Manual Box", command=self.add_manual_roi).pack(fill=tk.X, pady=4)

        frm_grid = ttk.Frame(self.control_panel, style="Main.TFrame")
        frm_grid.pack(fill=tk.X, pady=4)
        ttk.Label(frm_grid, text="Rows").pack(side=tk.LEFT)
        self.ent_rows = ttk.Entry(frm_grid, width=4)
        self.ent_rows.insert(0, "2")
        self.ent_rows.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(frm_grid, text="Cols").pack(side=tk.LEFT)
        self.ent_cols = ttk.Entry(frm_grid, width=4)
        self.ent_cols.insert(0, "2")
        self.ent_cols.pack(side=tk.LEFT, padx=(4, 0))

        self.var_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_panel, text="Grid Mode (drag to split)", variable=self.var_grid).pack(anchor="w", pady=6)

        ttk.Separator(self.control_panel).pack(fill="x", pady=8)

        # [3] ROI List
        ttk.Label(self.control_panel, text="[3] ROI List", font=TITLE_FONT).pack(anchor="w")
        self.roi_listbox = tk.Listbox(self.control_panel, height=8, font=MONO_FONT, bg="#0e1625", fg=TEXT_COLOR, 
                                      selectbackground="#1e3352", selectforeground=TEXT_COLOR, bd=0, 
                                      highlightthickness=1, highlightbackground="#1f2d46")
        self.roi_listbox.pack(fill=tk.X, pady=4)
        
        btn_frm = ttk.Frame(self.control_panel, style="Main.TFrame")
        btn_frm.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frm, text="Delete Selected", command=self.delete_roi).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_frm, text="Clear All", command=self.clear_all_rois).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # [4] Batch Export
        ttk.Label(self.control_panel, text="[4] Batch Export", font=TITLE_FONT).pack(anchor="w", pady=(12, 4))
        self.btn_process = ttk.Button(self.control_panel, text="Export ROI Stacks", style="Accent.TButton", 
                                      command=self.process_batch, state=tk.DISABLED)
        self.btn_process.pack(fill=tk.X, pady=(0, 12))

    def _build_viz_panel(self):
        ttk.Label(self.viz_panel, text="Alignment Progress", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.viz_panel, variable=self.progress_var, maximum=100, style="Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))
        self.lbl_progress = ttk.Label(self.viz_panel, text="Ready", font=("Menlo", 9), foreground=SUBTEXT_COLOR)
        self.lbl_progress.pack(fill=tk.X, pady=(0, 12))
        
        self.viz_fig = Figure(figsize=(5, 8), dpi=100, facecolor=PANEL_BG)
        self.viz_fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3)
        self.ax_lines = self.viz_fig.add_subplot(2, 1, 1)
        self.ax_plot = self.viz_fig.add_subplot(2, 1, 2)
        self.ax_lines.axis('off')
        self.ax_lines.set_facecolor(PANEL_BG)
        self.ax_plot.set_facecolor(PANEL_BG)
        self.ax_plot.tick_params(colors=TEXT_COLOR)
        for spine in self.ax_plot.spines.values(): spine.set_color(SUBTEXT_COLOR)
            
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.viz_panel)
        self.viz_canvas.draw()
        self.viz_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _maximize_window(self):
        try:
            self.root.state("zoomed")
        except:
            self.root.attributes("-zoomed", True)

    # ==================== FILE & ALIGNMENT ====================

    def load_folder(self):
        directory = filedialog.askdirectory(parent=self.root)
        if not directory: return
        
        valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        files = sorted([
            os.path.join(directory, f) for f in os.listdir(directory)
            if not f.startswith(".") and os.path.splitext(f)[1].lower() in valid_exts
            and os.path.isfile(os.path.join(directory, f))
        ])
        
        if not files:
            messagebox.showerror("Error", "No images found.")
            return
        
        self.folder_path = directory
        self.image_paths = files
        self.aligned_images = []
        self.aligned_images_basic = []
        self.aligned_images_final = []
        self.cv_image = None
        self.current_image_index = 0
        self.view_mode_var.set("Original")
        self._set_button_state(self.cmb_view, enable=False)
        self._load_current_image()
        self.clear_all_rois()
        self.display_image()
        self.lbl_status.config(text=f"Loaded {len(files)} frames.")
        self._set_button_state(self.btn_align, enable=True)

    def align_sequence(self):
        if not self.image_paths: return
        method = self.method_var.get()
        
        try:
            window = max(3, int(self.window_var.get()))
            max_dev = float(self.deviation_var.get())
            max_jump = float(self.jump_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters.")
            return
        
        self._set_button_state(self.btn_align, enable=False)
        self.progress_var.set(0)
        
        raw_angles = []
        cropped_images = []
        valid_paths = []
        
        # 1. Detect
        total = len(self.image_paths)
        for idx, path in enumerate(self.image_paths):
            self.lbl_progress.config(text=f"Phase 1/3: Detecting ({idx+1}/{total})")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            _, cropped = image_ops.remove_black_vignette(path)
            if cropped is None: continue
            
            angle, _, debug_viz = image_ops.get_deskew_angle_hough_gradient(cropped)
            raw_angles.append(angle)
            cropped_images.append(cropped)
            valid_paths.append(path)
            
            if idx % 2 == 0:
                self.ax_lines.clear()
                self.ax_lines.imshow(debug_viz)
                self.viz_canvas.draw()
            self.progress_var.set((idx+1)/total * 30)
            self.root.update()
            
        self.image_paths = valid_paths
        raw_angles_arr = np.array(raw_angles, dtype=float)

        # 2. Smooth
        self.lbl_progress.config(text="Phase 2/3: Smoothing...")
        if method == "temporal_smooth":
            final_angles_arr, outliers = image_ops.smooth_angles_temporal(raw_angles_arr, window, max_dev)
        elif method == "moving_average":
            final_angles_arr = image_ops.smooth_angles_moving_average(raw_angles_arr, window, max_jump)
            outliers = np.zeros_like(raw_angles_arr, dtype=bool)
        else:
            # Phase correlation placeholder for brevity
            final_angles_arr = raw_angles_arr
            outliers = np.zeros_like(raw_angles_arr, dtype=bool)

        # 3. Rotate
        aligned_images = []
        for idx, (img, angle) in enumerate(zip(cropped_images, final_angles_arr)):
            self.lbl_progress.config(text=f"Phase 3/3: Rotating ({idx+1}/{len(cropped_images)})")
            aligned_images.append(image_ops.rotate_image(img, angle))
            self.progress_var.set(30 + (idx+1)/len(cropped_images) * 70)
            self.root.update()
            
        self.aligned_images_basic = aligned_images
        self.aligned_images_final = list(aligned_images)
        self.aligned_images = self.aligned_images_basic
        self.raw_angles = raw_angles
        self.final_angles = final_angles_arr.tolist()
        self.corrected_frames = int(np.sum(outliers))
        self.alignment_ready = True
        
        self.view_mode_var.set("Aligned (Rotation)")
        self._set_button_state(self.cmb_view, enable=True)
        self._load_current_image()
        self.display_image()
        
        self._set_button_state(self.btn_align, enable=True)
        self._set_button_state(self.btn_plot, enable=True)
        self._set_button_state(self.btn_process, enable=True)
        self._set_button_state(self.btn_align_tops, enable=True)
        self.lbl_progress.config(text="Done.")

    def align_vertical_jitter(self):
        """Align images vertically based on the object's top edge."""
        if not self.alignment_ready or not self.aligned_images_basic:
            messagebox.showinfo("Align first", "Run the main alignment first.")
            return

        self._set_button_state(self.btn_align, enable=False)
        self._set_button_state(self.btn_align_tops, enable=False)
        self.progress_var.set(0)
        
        try:
            total = len(self.aligned_images_basic)
            
            try:
                v_window = int(self.vert_window_var.get())
                if v_window % 2 == 0: v_window += 1
            except ValueError:
                v_window = 1
            
            # Helper function to find the top edge
            def find_top_edge(img):
                # Use the center 50% of the image width to avoid edge artifacts
                h, w = img.shape
                center_strip = img[:, int(w*0.25):int(w*0.75)]
                profile = np.mean(center_strip, axis=1)
                
                # Simple smoothing to reduce noise
                kernel_size = 5
                kernel = np.ones(kernel_size) / kernel_size
                smooth_profile = np.convolve(profile, kernel, mode='same')
                
                # Dynamic threshold: 20% between min and max intensity
                mn, mx = np.min(smooth_profile), np.max(smooth_profile)
                threshold = mn + (mx - mn) * 0.2
                
                indices = np.where(smooth_profile > threshold)[0]
                if len(indices) > 0:
                    return indices[0], smooth_profile, threshold
                return 0, smooth_profile, threshold

            # 1. Collect all tops first
            raw_tops = []
            profiles = []
            thresholds = []
            
            for i, img in enumerate(self.aligned_images_basic):
                self.lbl_progress.config(text=f"Analyzing frame {i+1}/{total}")
                self.root.update()
                top, profile, thresh = find_top_edge(img)
                raw_tops.append(top)
                profiles.append(profile)
                thresholds.append(thresh)
                
                if i % 10 == 0: self.progress_var.set((i / total) * 30)
            
            # 2. Apply Median Filter
            raw_tops = np.array(raw_tops)
            if v_window > 1:
                self.lbl_progress.config(text=f"Applying median filter (window={v_window})...")
                self.root.update()
                # Simple 1D median filter
                smoothed_tops = np.copy(raw_tops)
                pad = v_window // 2
                for i in range(len(raw_tops)):
                    start = max(0, i - pad)
                    end = min(len(raw_tops), i + pad + 1)
                    smoothed_tops[i] = np.median(raw_tops[start:end])
            else:
                smoothed_tops = raw_tops

            # 3. Calculate Shifts
            ref_top = smoothed_tops[total // 2]
            shifts = []
            for top in smoothed_tops:
                if top == 0 and ref_top > 10:
                    shifts.append(0)
                else:
                    shifts.append(ref_top - top)
            
            # 4. Apply & Visualize
            new_aligned = []
            h, w = self.aligned_images_basic[0].shape
            
            for i, (img, shift) in enumerate(zip(self.aligned_images_basic, shifts)):
                self.lbl_progress.config(text=f"Applying vertical shifts ({i+1}/{total})")
                
                if shift != 0:
                    M = np.float32([[1, 0, 0], [0, 1, shift]])
                    shifted_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    new_aligned.append(shifted_img)
                else:
                    new_aligned.append(img)
                
                # Visual Debugging
                if i % 5 == 0 or i == total - 1:
                    self.ax_lines.clear()
                    self.ax_lines.plot(profiles[i], color=ACCENT_COLOR, label="Intensity Profile")
                    self.ax_lines.axhline(y=thresholds[i], color='gray', linestyle=':', alpha=0.5, label="Threshold")
                    self.ax_lines.axvline(x=raw_tops[i], color='red', linestyle='--', alpha=0.5, label=f"Raw Top: {raw_tops[i]}")
                    if v_window > 1:
                        self.ax_lines.axvline(x=smoothed_tops[i], color='#1dffd4', linestyle='-', alpha=0.8, label=f"Smooth Top: {int(smoothed_tops[i])}")
                    
                    self.ax_lines.set_title(f"Frame {i}: Top Edge Detection", color=TEXT_COLOR, fontsize=8)
                    self.ax_lines.tick_params(colors=TEXT_COLOR, labelsize=6)
                    self.ax_lines.grid(True, alpha=0.2)
                    self.ax_lines.legend(fontsize=6, loc='upper right', facecolor=PANEL_BG, labelcolor=TEXT_COLOR)
                    
                    self.ax_plot.clear()
                    self.ax_plot.plot(shifts, color=ROI_COLOR_SELECT)
                    self.ax_plot.set_title("Vertical Shift History", color=TEXT_COLOR, fontsize=8)
                    self.ax_plot.tick_params(colors=TEXT_COLOR, labelsize=6)
                    self.ax_plot.grid(True, alpha=0.2)
                    
                    self.viz_canvas.draw()
                    self.root.update()
                
                if i % 10 == 0:
                    self.progress_var.set(30 + (i / total) * 70)
            
            self.aligned_images_final = new_aligned
            self.aligned_images = self.aligned_images_final
            self.view_mode_var.set("Aligned (Full)")
            
            self._load_current_image()
            self.display_image()
            self.lbl_status.config(text=f"Vertical alignment complete. Adjusted tops for {total} frames.")
            self.lbl_progress.config(text="Vertical alignment complete.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Vertical alignment failed: {str(e)}")
            self.lbl_progress.config(text="Error in vertical alignment.")
        finally:
            self._set_button_state(self.btn_align, enable=True)
            self._set_button_state(self.btn_align_tops, enable=True)
            self.progress_var.set(100)

    def show_angle_plot(self):
        if not self.raw_angles: return
        win = tk.Toplevel(self.root)
        win.title("Angle Plot")
        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(self.raw_angles, label="Raw")
        ax.plot(self.final_angles, label="Smoothed")
        ax.legend()
        cv = FigureCanvasTkAgg(fig, win)
        cv.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        cv.draw()

    # ==================== DISPLAY & RESIZE LOGIC ====================

    def toggle_view_mode(self):
        mode = self.view_mode_var.get()
        if mode == "Original":
            self.aligned_images = []
        elif mode == "Aligned (Rotation)":
            self.aligned_images = self.aligned_images_basic
        elif mode == "Aligned (Full)":
            self.aligned_images = self.aligned_images_final
        self._load_current_image()
        self.display_image()

    def _load_current_image(self):
        mode = self.view_mode_var.get()
        if mode != "Original" and self.alignment_ready and self.aligned_images:
            if 0 <= self.current_image_index < len(self.aligned_images):
                self.cv_image = self.aligned_images[self.current_image_index]
        elif self.image_paths:
            if 0 <= self.current_image_index < len(self.image_paths):
                path = self.image_paths[self.current_image_index]
                self.cv_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def navigate_image(self, delta):
        count = len(self.aligned_images) if (self.alignment_ready and self.view_mode_var.get() != "Original") else len(self.image_paths)
        if count == 0: return
        self.current_image_index = max(0, min(self.current_image_index + delta, count - 1))
        self._load_current_image()
        self.display_image()
        self.lbl_status.config(text=f"Image {self.current_image_index + 1}/{count}")

    def on_canvas_resize(self, event):
        """Called whenever the canvas is resized by the user."""
        if self.cv_image is not None:
            self.display_image()

    def display_image(self):
        """Displays the image scaled to fit the CURRENT canvas size."""
        self.canvas.delete("all")
        if self.cv_image is None: return

        # Get current available space
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Guard against initialization size of 1x1
        if canvas_width < 10 or canvas_height < 10:
            return

        h, w = self.cv_image.shape
        
        # Calculate scale to fit ENTIRE image
        scale_w = canvas_width / w
        scale_h = canvas_height / h
        self.scale_factor = min(scale_w, scale_h)
        
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        resized = cv2.resize(self.cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.current_display_img = ImageTk.PhotoImage(Image.fromarray(resized))
        
        # Center the image
        x_offset = (canvas_width - new_w) // 2
        y_offset = (canvas_height - new_h) // 2
        
        self.canvas.create_image(x_offset, y_offset, image=self.current_display_img, anchor="nw", tags="bg")
        
        # Offset ROIs by the centering amount
        self.roi_offset_x = x_offset
        self.roi_offset_y = y_offset
        
        self.redraw_rois()

    # ==================== ROI INTERACTION ====================
    # (Mouse handlers modified to account for centering offset)

    def on_mouse_down(self, event):
        if self.cv_image is None: return
        adj_x = event.x - getattr(self, 'roi_offset_x', 0)
        adj_y = event.y - getattr(self, 'roi_offset_y', 0)
        
        # Logic for starting draw or selecting ROI (simplified)
        self.drag_mode = "DRAWING"
        self.start_x = adj_x
        self.start_y = adj_y
        self.temp_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, 
            outline=ROI_COLOR_SELECT, dash=(6,4), tags="temp"
        )

    def on_mouse_drag(self, event):
        if self.drag_mode == "DRAWING":
            # Update visual rectangle
            orig_start_x = self.start_x + getattr(self, 'roi_offset_x', 0)
            orig_start_y = self.start_y + getattr(self, 'roi_offset_y', 0)
            self.canvas.coords(self.temp_rect_id, orig_start_x, orig_start_y, event.x, event.y)
            
            # Update entry fields with scaled coords
            cur_x = event.x - getattr(self, 'roi_offset_x', 0)
            cur_y = event.y - getattr(self, 'roi_offset_y', 0)
            w = abs(cur_x - self.start_x) / self.scale_factor
            h = abs(cur_y - self.start_y) / self.scale_factor
            self._fill_inputs(0, 0, w, h)

    def on_mouse_up(self, event):
        if self.drag_mode == "DRAWING":
            self.canvas.delete(self.temp_rect_id)
            adj_x = event.x - getattr(self, 'roi_offset_x', 0)
            adj_y = event.y - getattr(self, 'roi_offset_y', 0)
            
            x1, y1 = self.start_x, self.start_y
            x2, y2 = adj_x, adj_y
            
            rx = int(min(x1, x2) / self.scale_factor)
            ry = int(min(y1, y2) / self.scale_factor)
            rw = int(abs(x2 - x1) / self.scale_factor)
            rh = int(abs(y2 - y1) / self.scale_factor)
            
            if rw > MIN_ROI_SIZE and rh > MIN_ROI_SIZE:
                if self.var_grid.get():
                    self.split_grid(rx, ry, rw, rh)
                else:
                    self.add_roi_data(rx, ry, rw, rh)
        self.drag_mode = None

    def add_roi_data(self, x, y, w, h, tag=None, refresh=True):
        if tag is None: tag = f"ROI_{len(self.rois)}"
        roi = {'x': x, 'y': y, 'w': w, 'h': h, 'tag': tag}
        self.rois.append(roi)
        if refresh:
            self.update_listbox()
            self.redraw_rois()
        return roi
    
    def add_manual_roi(self):
        try:
            v = [int(self.entries[k].get()) for k in ["X","Y","W","H"]]
            self.add_roi_data(*v)
        except: pass

    def split_grid(self, x, y, w, h):
        r, c = int(self.ent_rows.get()), int(self.ent_cols.get())
        for i in range(r):
            for j in range(c):
                nx = x + int(j * w/c)
                ny = y + int(i * h/r)
                nw = int(w/c)
                nh = int(h/r)
                self.add_roi_data(nx, ny, nw, nh, refresh=False)
        self.update_listbox()
        self.redraw_rois()

    def delete_roi(self):
        sel = self.roi_listbox.curselection()
        if sel:
            self.rois.pop(sel[0])
            self.update_listbox()
            self.redraw_rois()

    def clear_all_rois(self):
        self.rois = []
        self.update_listbox()
        self.redraw_rois()

    def update_listbox(self):
        self.roi_listbox.delete(0, tk.END)
        for roi in self.rois:
            self.roi_listbox.insert(tk.END, f"{roi['tag']} {roi['w']}x{roi['h']}")

    def redraw_rois(self):
        self.canvas.delete("roi")
        off_x = getattr(self, 'roi_offset_x', 0)
        off_y = getattr(self, 'roi_offset_y', 0)
        
        for roi in self.rois:
            sx = roi['x'] * self.scale_factor + off_x
            sy = roi['y'] * self.scale_factor + off_y
            sw = roi['w'] * self.scale_factor
            sh = roi['h'] * self.scale_factor
            self.canvas.create_rectangle(sx, sy, sx+sw, sy+sh, outline=ROI_COLOR_NORMAL, width=2, tags="roi")
            self.canvas.create_text(sx, sy-10, text=roi['tag'], fill=ROI_COLOR_NORMAL, anchor="sw", tags="roi")

    def _fill_inputs(self, x, y, w, h):
        for k, v in zip(["X","Y","W","H"], [x,y,w,h]):
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, str(int(v)))

    def process_batch(self):
        """Process and export all ROIs across the aligned sequence."""
        if not self.rois:
            messagebox.showinfo("No ROIs", "Define at least one ROI before exporting.")
            return
        
        # Determine which set of images to use based on view mode
        mode = self.view_mode_var.get()
        export_images = []
        using_files = False
        
        if mode == "Original":
            if not self.image_paths:
                messagebox.showinfo("No data", "No images loaded.")
                return
            using_files = True
            total_frames = len(self.image_paths)
            
        elif mode == "Aligned (Rotation)":
            if not self.aligned_images_basic:
                messagebox.showinfo("No data", "No rotation-aligned images found. Run 'Align Volume' first.")
                return
            export_images = self.aligned_images_basic
            total_frames = len(export_images)
            
        elif mode == "Aligned (Full)":
            if not self.aligned_images_final:
                # Fallback if they selected Full but haven't run it
                if self.aligned_images_basic:
                    messagebox.showwarning("Warning", "Full alignment not ready. Falling back to Rotation alignment.")
                    export_images = self.aligned_images_basic
                    total_frames = len(export_images)
                else:
                    messagebox.showinfo("No data", "No aligned images found.")
                    return
            else:
                export_images = self.aligned_images_final
                total_frames = len(export_images)
        
        out_dir = filedialog.askdirectory()
        if not out_dir:
            return
        
        self._set_button_state(self.btn_process, enable=False)
        self.progress_var.set(0)
        
        try:
            roi_stacks = {roi['tag']: [] for roi in self.rois}
            
            for i in range(total_frames):
                self.lbl_progress.config(text=f"Exporting ({mode}): Processing frame {i + 1}/{total_frames}")
                self.root.update()
                
                # Get image source
                if using_files:
                    img = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                else:
                    img = export_images[i]
                
                # Extract ROIs
                for roi in self.rois:
                    # Ensure coordinates are within bounds
                    y2 = min(roi['y'] + roi['h'], img.shape[0])
                    x2 = min(roi['x'] + roi['w'], img.shape[1])
                    
                    # Handle edge cases where ROI might be outside image
                    if roi['y'] >= img.shape[0] or roi['x'] >= img.shape[1]:
                        continue
                        
                    crop = img[roi['y']:y2, roi['x']:x2]
                    
                    if crop.size == 0:
                        continue
                    
                    roi_stacks[roi['tag']].append(Image.fromarray(crop))
                
                if i % 5 == 0:
                    self.progress_var.set((i / total_frames) * 80)
            
            self.lbl_progress.config(text="Exporting: Writing TIFF stacks...")
            self.root.update()
            
            saved = 0
            total_rois = len(roi_stacks)
            for idx, (tag, frames) in enumerate(roi_stacks.items()):
                if frames:
                    save_path = os.path.join(out_dir, f"{tag}_{mode.replace(' ', '_')}_stack.tif")
                    frames[0].save(save_path, save_all=True, append_images=frames[1:], compression="tiff_deflate")
                    saved += 1
                
                self.progress_var.set(80 + ((idx + 1) / total_rois) * 20)
                self.root.update()
            
            messagebox.showinfo("Export complete", f"Saved {saved} ROI stacks to {out_dir}")
            self.lbl_status.config(text=f"Export complete. Saved {saved} stacks.")
            self.lbl_progress.config(text="Export complete.")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred: {str(e)}")
            self.lbl_status.config(text="Export failed.")
            self.lbl_progress.config(text="Export failed.")
            
        finally:
            self._set_button_state(self.btn_process, enable=True)
            self.progress_var.set(100)

    def _set_button_state(self, btn, enable=True):
        state = "!disabled" if enable else "disabled"
        btn.state([state])

if __name__ == "__main__":
    root = tk.Tk()
    app = SubvolumeApp(root)
    root.mainloop()