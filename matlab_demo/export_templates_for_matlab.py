"""
Export templates and recordings to .mat format for MATLAB demo.

This script provides a GUI to select and export:
1. Template sets (from data/templates/<folder>/)
2. Recordings (from data/recordings/<date>/)

Output files will be saved in matlab_demo/data/

Usage:
    python export_templates_for_matlab.py
"""

import os
import pickle
import numpy as np
from scipy.io import savemat
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mindmove.config import config
from mindmove.model.core.windowing import sliding_window
from mindmove.model.core.features.time_domain import compute_waveform_length, compute_rms


class ExportGUI:
    """GUI for selecting and exporting templates/recordings/models to MATLAB."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Export to MATLAB")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        # Data paths
        self.templates_dir = Path("data/templates")
        self.recordings_dir = Path("data/recordings")
        self.models_dir = Path("data/models")
        self.output_dir = Path("matlab_demo/data")

        self._setup_ui()
        self._refresh_lists()

    def _setup_ui(self):
        """Setup the GUI layout."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)

        # === TEMPLATES SECTION ===
        templates_frame = ttk.LabelFrame(main_frame, text="Template Sets", padding="5")
        templates_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        templates_frame.columnconfigure(0, weight=1)
        templates_frame.rowconfigure(1, weight=1)

        ttk.Label(templates_frame, text="Select template folders to export:").grid(row=0, column=0, sticky="w")

        # Templates listbox with scrollbar
        templates_list_frame = ttk.Frame(templates_frame)
        templates_list_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        templates_list_frame.columnconfigure(0, weight=1)
        templates_list_frame.rowconfigure(0, weight=1)

        self.templates_listbox = tk.Listbox(templates_list_frame, selectmode=tk.EXTENDED, height=10)
        templates_scrollbar = ttk.Scrollbar(templates_list_frame, orient="vertical", command=self.templates_listbox.yview)
        self.templates_listbox.configure(yscrollcommand=templates_scrollbar.set)
        self.templates_listbox.grid(row=0, column=0, sticky="nsew")
        templates_scrollbar.grid(row=0, column=1, sticky="ns")

        # Templates buttons
        templates_btn_frame = ttk.Frame(templates_frame)
        templates_btn_frame.grid(row=2, column=0, sticky="ew")
        ttk.Button(templates_btn_frame, text="Select All", command=self._select_all_templates).pack(side="left", padx=2)
        ttk.Button(templates_btn_frame, text="Clear", command=self._clear_templates).pack(side="left", padx=2)

        # === RECORDINGS SECTION ===
        recordings_frame = ttk.LabelFrame(main_frame, text="Recordings", padding="5")
        recordings_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        recordings_frame.columnconfigure(0, weight=1)
        recordings_frame.rowconfigure(1, weight=1)

        ttk.Label(recordings_frame, text="Select recordings to export:").grid(row=0, column=0, sticky="w")

        # Recordings treeview with scrollbar (hierarchical: folder > files)
        recordings_tree_frame = ttk.Frame(recordings_frame)
        recordings_tree_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        recordings_tree_frame.columnconfigure(0, weight=1)
        recordings_tree_frame.rowconfigure(0, weight=1)

        self.recordings_tree = ttk.Treeview(recordings_tree_frame, selectmode="extended", height=10)
        recordings_scrollbar = ttk.Scrollbar(recordings_tree_frame, orient="vertical", command=self.recordings_tree.yview)
        self.recordings_tree.configure(yscrollcommand=recordings_scrollbar.set)
        self.recordings_tree.grid(row=0, column=0, sticky="nsew")
        recordings_scrollbar.grid(row=0, column=1, sticky="ns")

        self.recordings_tree.heading("#0", text="Recordings", anchor="w")

        # Recordings buttons
        recordings_btn_frame = ttk.Frame(recordings_frame)
        recordings_btn_frame.grid(row=2, column=0, sticky="ew")
        ttk.Button(recordings_btn_frame, text="Select All", command=self._select_all_recordings).pack(side="left", padx=2)
        ttk.Button(recordings_btn_frame, text="Clear", command=self._clear_recordings).pack(side="left", padx=2)
        ttk.Button(recordings_btn_frame, text="Expand All", command=self._expand_all).pack(side="left", padx=2)

        # === MODELS SECTION ===
        models_frame = ttk.LabelFrame(main_frame, text="Trained Models", padding="5")
        models_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        models_frame.columnconfigure(0, weight=1)
        models_frame.rowconfigure(1, weight=1)

        ttk.Label(models_frame, text="Select models to export:").grid(row=0, column=0, sticky="w")

        # Models listbox with scrollbar
        models_list_frame = ttk.Frame(models_frame)
        models_list_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        models_list_frame.columnconfigure(0, weight=1)
        models_list_frame.rowconfigure(0, weight=1)

        self.models_listbox = tk.Listbox(models_list_frame, selectmode=tk.EXTENDED, height=10)
        models_scrollbar = ttk.Scrollbar(models_list_frame, orient="vertical", command=self.models_listbox.yview)
        self.models_listbox.configure(yscrollcommand=models_scrollbar.set)
        self.models_listbox.grid(row=0, column=0, sticky="nsew")
        models_scrollbar.grid(row=0, column=1, sticky="ns")

        # Models buttons
        models_btn_frame = ttk.Frame(models_frame)
        models_btn_frame.grid(row=2, column=0, sticky="ew")
        ttk.Button(models_btn_frame, text="Select All", command=self._select_all_models).pack(side="left", padx=2)
        ttk.Button(models_btn_frame, text="Clear", command=self._clear_models).pack(side="left", padx=2)

        # === OUTPUT SECTION ===
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="5")
        output_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output folder:").grid(row=0, column=0, sticky="w", padx=5)
        self.output_var = tk.StringVar(value=str(self.output_dir))
        output_entry = ttk.Entry(output_frame, textvariable=self.output_var)
        output_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(output_frame, text="Browse...", command=self._browse_output).grid(row=0, column=2, padx=5)

        # === EXPORT OPTIONS ===
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        self.export_raw_var = tk.BooleanVar(value=True)
        self.export_features_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(options_frame, text="Export raw EMG data", variable=self.export_raw_var).pack(side="left", padx=10)
        ttk.Checkbutton(options_frame, text="Export feature-extracted data", variable=self.export_features_var).pack(side="left", padx=10)

        # === ACTION BUTTONS ===
        action_frame = ttk.Frame(main_frame)
        action_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Button(action_frame, text="Refresh", command=self._refresh_lists).pack(side="left", padx=5)
        ttk.Button(action_frame, text="Export Selected", command=self._export_selected).pack(side="right", padx=5)

        # === STATUS ===
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_label.grid(row=4, column=0, columnspan=3, sticky="ew", pady=5)

        # Configure row weights
        main_frame.rowconfigure(0, weight=1)

    def _refresh_lists(self):
        """Refresh the template, recording, and model lists."""
        # Clear existing items
        self.templates_listbox.delete(0, tk.END)
        self.models_listbox.delete(0, tk.END)
        for item in self.recordings_tree.get_children():
            self.recordings_tree.delete(item)

        # Populate templates
        if self.templates_dir.exists():
            for folder in sorted(self.templates_dir.iterdir()):
                if folder.is_dir():
                    # Check if it contains template files
                    open_pkl = folder / "templates_open.pkl"
                    closed_pkl = folder / "templates_closed.pkl"
                    if open_pkl.exists() or closed_pkl.exists():
                        self.templates_listbox.insert(tk.END, folder.name)

        # Populate recordings (hierarchical)
        if self.recordings_dir.exists():
            for subfolder in sorted(self.recordings_dir.iterdir()):
                if subfolder.is_dir():
                    # Add folder node
                    folder_id = self.recordings_tree.insert("", "end", text=f"📁 {subfolder.name}", open=False)

                    # Add recording files in folder
                    for recording in sorted(subfolder.glob("*.pkl")):
                        self.recordings_tree.insert(folder_id, "end", text=f"📄 {recording.name}",
                                                   values=(str(recording),))
                elif subfolder.suffix == ".pkl":
                    # Direct recording file in root
                    self.recordings_tree.insert("", "end", text=f"📄 {subfolder.name}",
                                               values=(str(subfolder),))

        # Populate models
        if self.models_dir.exists():
            for model_file in sorted(self.models_dir.glob("*.pkl")):
                self.models_listbox.insert(tk.END, model_file.name)

        self.status_var.set(f"Found {self.templates_listbox.size()} template sets, "
                          f"{len(self._get_all_recording_files())} recordings, "
                          f"{self.models_listbox.size()} models")

    def _get_all_recording_files(self):
        """Get all recording file paths from the tree."""
        files = []
        def traverse(item):
            values = self.recordings_tree.item(item, "values")
            if values and values[0]:
                files.append(values[0])
            for child in self.recordings_tree.get_children(item):
                traverse(child)

        for item in self.recordings_tree.get_children():
            traverse(item)
        return files

    def _select_all_templates(self):
        self.templates_listbox.select_set(0, tk.END)

    def _clear_templates(self):
        self.templates_listbox.selection_clear(0, tk.END)

    def _select_all_recordings(self):
        def select_all(item):
            self.recordings_tree.selection_add(item)
            for child in self.recordings_tree.get_children(item):
                select_all(child)

        for item in self.recordings_tree.get_children():
            select_all(item)

    def _clear_recordings(self):
        self.recordings_tree.selection_remove(*self.recordings_tree.selection())

    def _select_all_models(self):
        self.models_listbox.select_set(0, tk.END)

    def _clear_models(self):
        self.models_listbox.selection_clear(0, tk.END)

    def _expand_all(self):
        def expand(item):
            self.recordings_tree.item(item, open=True)
            for child in self.recordings_tree.get_children(item):
                expand(child)

        for item in self.recordings_tree.get_children():
            expand(item)

    def _browse_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_var.get())
        if folder:
            self.output_var.set(folder)

    def _export_selected(self):
        """Export selected templates and recordings."""
        output_dir = Path(self.output_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        errors = []

        # Export selected templates
        selected_templates = [self.templates_listbox.get(i) for i in self.templates_listbox.curselection()]
        for template_name in selected_templates:
            try:
                self._export_template_set(template_name, output_dir)
                exported_count += 1
            except Exception as e:
                errors.append(f"Template '{template_name}': {e}")

        # Export selected recordings
        selected_items = self.recordings_tree.selection()
        recording_files = set()

        for item in selected_items:
            values = self.recordings_tree.item(item, "values")
            if values and values[0]:
                # It's a file
                recording_files.add(values[0])
            else:
                # It's a folder - get all children
                for child in self.recordings_tree.get_children(item):
                    child_values = self.recordings_tree.item(child, "values")
                    if child_values and child_values[0]:
                        recording_files.add(child_values[0])

        for recording_path in recording_files:
            try:
                self._export_recording(recording_path, output_dir)
                exported_count += 1
            except Exception as e:
                errors.append(f"Recording '{Path(recording_path).name}': {e}")

        # Export selected models
        selected_models = [self.models_listbox.get(i) for i in self.models_listbox.curselection()]
        for model_name in selected_models:
            try:
                self._export_model(model_name, output_dir)
                exported_count += 1
            except Exception as e:
                errors.append(f"Model '{model_name}': {e}")

        # Show result
        if errors:
            messagebox.showwarning("Export Complete with Errors",
                                  f"Exported {exported_count} items.\n\nErrors:\n" + "\n".join(errors[:5]))
        elif exported_count > 0:
            messagebox.showinfo("Export Complete", f"Successfully exported {exported_count} items to:\n{output_dir}")
        else:
            messagebox.showinfo("Nothing Selected", "Please select templates or recordings to export.")

        self.status_var.set(f"Exported {exported_count} items")

    def _export_template_set(self, folder_name: str, output_dir: Path):
        """Export a template set folder."""
        folder_path = self.templates_dir / folder_name

        templates_open_raw = []
        templates_closed_raw = []
        templates_open_features = []
        templates_closed_features = []

        # Load OPEN templates
        open_pkl = folder_path / "templates_open.pkl"
        if open_pkl.exists():
            with open(open_pkl, 'rb') as f:
                templates = pickle.load(f)
                if isinstance(templates, list):
                    for t in templates:
                        if isinstance(t, np.ndarray):
                            templates_open_raw.append(t)
                            if self.export_features_var.get():
                                windowed = sliding_window(t, config.window_length, config.increment)
                                templates_open_features.append(compute_rms(windowed))
                elif isinstance(templates, np.ndarray):
                    templates_open_raw.append(templates)
                    if self.export_features_var.get():
                        windowed = sliding_window(templates, config.window_length, config.increment)
                        templates_open_features.append(compute_rms(windowed))

        # Load CLOSED templates
        closed_pkl = folder_path / "templates_closed.pkl"
        if closed_pkl.exists():
            with open(closed_pkl, 'rb') as f:
                templates = pickle.load(f)
                if isinstance(templates, list):
                    for t in templates:
                        if isinstance(t, np.ndarray):
                            templates_closed_raw.append(t)
                            if self.export_features_var.get():
                                windowed = sliding_window(t, config.window_length, config.increment)
                                templates_closed_features.append(compute_rms(windowed))
                elif isinstance(templates, np.ndarray):
                    templates_closed_raw.append(templates)
                    if self.export_features_var.get():
                        windowed = sliding_window(templates, config.window_length, config.increment)
                        templates_closed_features.append(compute_rms(windowed))

        # Build mat data
        mat_data = {
            'Fs': config.FSAMP,
            'window_size': config.window_length,
            'window_shift': config.increment,
            'template_duration_s': config.template_duration,
            'n_channels': config.num_channels,
            'folder_name': folder_name,
        }

        if self.export_raw_var.get():
            mat_data['templates_open_raw'] = np.array(templates_open_raw) if templates_open_raw else np.array([])
            mat_data['templates_closed_raw'] = np.array(templates_closed_raw) if templates_closed_raw else np.array([])

        if self.export_features_var.get():
            mat_data['templates_open_features'] = np.array(templates_open_features) if templates_open_features else np.array([])
            mat_data['templates_closed_features'] = np.array(templates_closed_features) if templates_closed_features else np.array([])

        # Save
        output_path = output_dir / f"templates_{folder_name}.mat"
        savemat(str(output_path), mat_data)

        print(f"Exported template set '{folder_name}':")
        print(f"  OPEN:   {len(templates_open_raw)} templates")
        print(f"  CLOSED: {len(templates_closed_raw)} templates")
        print(f"  Output: {output_path}")

    def _export_recording(self, recording_path: str, output_dir: Path):
        """Export a single recording to .mat format."""
        recording_path = Path(recording_path)

        with open(recording_path, 'rb') as f:
            recording = pickle.load(f)

        # Extract data
        emg = recording.get('emg', recording.get('biosignal'))
        gt = recording.get('gt', recording.get('ground_truth', recording.get('kinematics')))

        if gt is not None and gt.ndim > 1:
            gt = gt.flatten()

        mat_data = {
            'emg': emg,
            'gt': gt,
            'Fs': config.FSAMP,
            'label': recording.get('label', 'unknown'),
        }

        # Add additional metadata if available
        if 'animation_config' in recording:
            anim_config = recording['animation_config']
            mat_data['hold_open_s'] = anim_config.get('hold_open_s', 0)
            mat_data['hold_closed_s'] = anim_config.get('hold_closed_s', 0)
            mat_data['closing_s'] = anim_config.get('closing_s', 0)
            mat_data['opening_s'] = anim_config.get('opening_s', 0)
            mat_data['protocol_mode'] = anim_config.get('protocol_mode', 'standard')

        if 'close_cue_time' in recording:
            mat_data['close_cue_time'] = recording['close_cue_time']
        if 'open_cue_time' in recording:
            mat_data['open_cue_time'] = recording['open_cue_time']

        # Create output filename (include parent folder name to avoid conflicts)
        parent_name = recording_path.parent.name
        base_name = recording_path.stem
        output_path = output_dir / f"recording_{parent_name}_{base_name}.mat"

        savemat(str(output_path), mat_data)

        print(f"Exported recording: {recording_path.name}")
        print(f"  EMG shape: {emg.shape if emg is not None else 'None'}")
        print(f"  GT shape: {gt.shape if gt is not None else 'None'}")
        print(f"  Output: {output_path}")

    def _export_model(self, model_name: str, output_dir: Path):
        """Export a trained model to .mat format."""
        model_path = self.models_dir / model_name

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Extract templates - these are already feature-extracted in models
        # Shape is typically (n_windows, n_channels) e.g. (29, 32)
        open_templates = model.get('open_templates', [])
        closed_templates = model.get('closed_templates', [])

        # Templates in models are already features, not raw EMG
        open_templates_features = []
        closed_templates_features = []

        for t in open_templates:
            if isinstance(t, np.ndarray):
                open_templates_features.append(t)

        for t in closed_templates:
            if isinstance(t, np.ndarray):
                closed_templates_features.append(t)

        # Build mat data
        mat_data = {
            # Parameters
            'Fs': config.FSAMP,
            'window_size': config.window_length,
            'window_shift': config.increment,
            'n_channels': config.num_channels,

            # Thresholds
            'threshold_base_open': model.get('threshold_base_open', 0),
            'threshold_base_closed': model.get('threshold_base_closed', 0),
            'mean_open': model.get('mean_open', 0),
            'std_open': model.get('std_open', 0),
            'mean_closed': model.get('mean_closed', 0),
            'std_closed': model.get('std_closed', 0),
            'mean_cross': model.get('mean_cross', 0),
            'std_cross': model.get('std_cross', 0),

            # Model settings
            'feature_name': model.get('feature_name', 'unknown'),
            'distance_aggregation': model.get('distance_aggregation', 'average'),
            'smoothing_method': model.get('smoothing_method', 'none'),
            'differential_mode': str(model.get('differential_mode', False)),

            # Dead channels
            'dead_channels': np.array(model.get('dead_channels', [])),

            # Templates (already feature-extracted in models)
            'templates_open_features': np.array(open_templates_features) if open_templates_features else np.array([]),
            'templates_closed_features': np.array(closed_templates_features) if closed_templates_features else np.array([]),
        }

        # Add threshold presets if available
        if 'threshold_presets' in model and model['threshold_presets']:
            presets = model['threshold_presets']
            preset_names = list(presets.keys())
            mat_data['threshold_preset_names'] = preset_names

            # Handle different preset structures
            preset_open = []
            preset_closed = []
            for name in preset_names:
                p = presets[name]
                # New format: threshold_open/threshold_closed
                if 'threshold_open' in p:
                    preset_open.append(p['threshold_open'])
                    preset_closed.append(p['threshold_closed'])
                # Old format: open/closed directly
                elif 'open' in p:
                    preset_open.append(p['open'])
                    preset_closed.append(p['closed'])

            if preset_open:
                mat_data['threshold_presets_open'] = np.array(preset_open)
                mat_data['threshold_presets_closed'] = np.array(preset_closed)

        # Add metadata if available
        if 'metadata' in model:
            meta = model['metadata']
            mat_data['model_name'] = meta.get('model_name', model_name)
            mat_data['created_at'] = meta.get('created_at', 'unknown')
            mat_data['n_open_templates'] = meta.get('n_open_templates', len(open_templates))
            mat_data['n_closed_templates'] = meta.get('n_closed_templates', len(closed_templates))

        # Save
        base_name = Path(model_name).stem
        output_path = output_dir / f"model_{base_name}.mat"
        savemat(str(output_path), mat_data)

        print(f"Exported model: {model_name}")
        print(f"  OPEN templates:   {len(open_templates_features)}")
        print(f"  CLOSED templates: {len(closed_templates_features)}")
        print(f"  Threshold OPEN:   {mat_data['threshold_base_open']:.4f}")
        print(f"  Threshold CLOSED: {mat_data['threshold_base_closed']:.4f}")
        print(f"  Output: {output_path}")

    def run(self):
        """Start the GUI."""
        self.root.mainloop()


def main():
    """Main entry point."""
    print("=== Export to MATLAB ===")
    print("Opening selection window...\n")

    gui = ExportGUI()
    gui.run()


if __name__ == "__main__":
    main()
