# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Robustic** is built on top of MindMove, a Python framework for real-time EMG signal processing. It implements a **1-DoF hand state classifier** (OPEN / CLOSED) designed for robustness against Activities of Daily Living (ADL), targeting prosthetic hand control for SCI patients. Uses DTW-based template matching with cosine distance, aiming for <50ms decision latency.

The algorithm has two key layers:
1. **Cosine DTW** — standard DTW dynamic programming but with cosine distance as local cost (`1 - cos(t1[i], t2[j])`) instead of Euclidean. Amplitude-invariant, direction-sensitive.
2. **Spatial correction** — validates that the right muscles are active by comparing RMS channel activation profiles (cosine similarity). Available in two modes: global (post-aggregation) and per-template coupled (pre-aggregation).

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python mindmove/main.py

# Run tests
uv run pytest mindmove/model/tests/

# Run a single test file
uv run pytest mindmove/model/tests/test_model_training.py

# Run DTW offline testing tool
uv run python dtw_offline_test.py --no-gui

# Compile Qt Designer .ui files after editing
pyside6-uic mindmove/gui/ui/file.ui -o mindmove/gui/ui_compiled/file.py

# Install PyTorch with CUDA (optional)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Python >= 3.11 required. Key pinned deps: PySide6 6.7.0, numpy 1.26.4, vispy 0.14.2, lightning 2.2.2.

## Architecture

### Signal Flow Pipeline
```
Muovi Device (TCP/IP, 32-ch EMG, 2000 Hz)
       |
       v
MuoviWidget.extract_emg_data()
  - Differential mode (optional): 32 ch → 16 ch single differential
  - Real-time filtering (if enabled): bandpass + notch
  - Filter caching prevents double-filtering same packet
       |
       v
ready_read_signal (Qt Signal)
       |
       +---> Real-time Plot (VisPy)
       |
       +---> Protocol (Record / GuidedRecord / Training / Online)
                    |
                    v
              Model.predict()  <-- DTW computation every 50ms
                    |
                    v
              VirtualHandInterface (UDP to Unity: [state, distance, threshold])
```

### Four Operating Modes (Protocols)

1. **RecordProtocol** (`protocols/record.py`) — Basic raw EMG + kinematics capture
2. **GuidedRecordProtocol** (`protocols/guided_record.py`) — Patient-friendly template recording with VHI animation guide. VHI shows timed open→close→open animation; patient follows the virtual hand. GT derived from VHI animation state. Therapist controls timing between cycles. Audio cues signal transitions. Supports configurable repetitions per cycle and reject-last-cycle functionality.
3. **TrainingProtocol** (`protocols/training.py`) — Dataset creation from recordings, template extraction with individual accept/reject per class (Accept Both / Accept CLOSED / Accept OPEN), threshold computation, model saving. Template extraction supports: GT-based, audio-cue, onset detection (amplitude or TKEO). Multi-repetition recordings supported. Template Study dialog for quality analysis. Decision model training (CatBoost/NN).
4. **OnlineProtocol** (`protocols/online.py`) — Real-time prediction with calibration worker, sends `[state, distance, threshold]` to Unity via UDP (0.0=OPEN, 1.0=CLOSED). Decision model selector (Threshold/CatBoost/NN). Spatial correction controls: mode, profile type (mean/per-template top-k), coupling (global/per-template), threshold/sharpness/baseline sliders. Offline simulation with GT comparison. All settings apply live during prediction (main Qt thread, model reads attrs fresh each predict() call).

### Key Layers

**GUI** (`mindmove/gui/`): Main window (`mindmove.py`), protocol manager (`protocol.py`), protocol implementations (`protocols/`), custom widgets (`widgets/`), virtual hand UDP interface (`virtual_hand_interface.py`). UI files in `gui/ui/` (Qt Designer), compiled to `gui/ui_compiled/`. Signal Processing group box with filter/differential toggles is created programmatically in `muovi_widget.py`.

**Device** (`mindmove/device_interfaces/`): `BaseDevice` abstract class with Qt signals (`device.py`), Muovi TCP/IP implementation (`muovi.py`), device UI widget (`gui/muovi_widget.py`)

**Model** (`mindmove/model/`): `MindMoveInterface` facade (`interface.py`) — always use this, not Model directly. DTW state machine classifier (`core/model.py`), algorithm implementations (`core/algorithm.py`), filtering (`core/filtering.py`), feature extraction (`core/features/`), template management (`templates/`), decision models (`core/decision_network.py`), offline simulation (`offline_test.py`)

### DTW Classifier

**State machine**: Computes DTW distance to opposite-state templates. If distance < threshold → switch state. Then applies smoothing (majority vote or N consecutive).

**Threshold computation** — multiple approaches in `algorithm.py`:
- `compute_threshold()` — Intra-class: `threshold = mean + s * std` (tunable via UI sliders)
- `compute_threshold_presets()` — 4 presets: Current (intra-class), Cross-class Midpoint, Conservative (no false triggers), Safety Margin (50%)
- `find_plateau_thresholds()` — Analyzes state plateaus in calibration recording
- `find_transition_based_thresholds()` — Focuses on GT transition neighborhoods; finds minimum distance to target class around each transition; `threshold = mean_of_mins + k * std` (newer approach, not yet integrated into calibration)

**DTW implementations** (selected via config flags):
- `dtw_numba()` — Numba JIT, cosine distance (default, fastest CPU)
- `dtw()` — Pure Python, cosine distance
- `dtw_tslearn()` — tslearn library, Euclidean distance
- `dtw_gpudtw()` — GPU-accelerated (requires CUDA/OpenCL)

**Distance aggregation**: `average` (default), `minimum`, `avg_3_smallest`

**Spatial correction** (`algorithm.py`):
- `compute_spatial_profiles()` — builds L2-normalized RMS vectors per template (second half of each 1s window). Also computes mean profile, max-norm profile (for plotting), consistency weights (`mean²/(mean+std)`)
- `compute_spatial_similarity()` — runtime cosine sim between live RMS and class mean profile
- `compute_live_rms()` — L2-normalized RMS of second half of live buffer
- `aggregate_distances_with_per_template_spatial()` — per-template coupled mode: corrects each template distance individually before aggregation
- Correction modes: gate, scaling (`D/sim^k`), relu_scaling, relu_ext_scaling, contrast (`D * (sim_current/sim_target)^k`)
- Per-template coupling: only gate/scaling/relu modes (contrast needs two class sims per sample, no per-template equivalent)

**Decision models** (`core/decision_network.py`):
- **Threshold mode** (default): state machine with DTW distance vs threshold comparison
- **CatBoost transition mode**: `predict_transition(D_target, sim_target)` — state-conditioned, features `[D_target, sim_target, D/sim]`. Immune to ADL D_ratio inflation that plagued old posture-classifier.
- **Neural Network**: Lightning-based, same transition interface
- Decision model selection in both training (None/CatBoost/NN/Both) and online (Threshold/CatBoost/NN dropdown)

**Template Study** (`training.py: TemplateStudyDialog`):
- Tab 1: DTW distance matrix heatmap + intra/inter class statistics + grade (0-10)
- Tab 2: Spatial profile analysis — per-template RMS heatmap, mean profile bars, consistency weights
- Tab 3: Coupled correction preview — corrected distance matrix, DTW-vs-spatial scatter, stats comparison with raw

**Template provenance**: Each template stores metadata `{"id": stable_int, "recording": str, "cycle": int}`. Stable IDs survive removals (C39 stays C39 if C38 deleted). Displayed in Template Study labels and flagged-templates list.

### Signal Processing

Filtering happens at source level in `MuoviWidget.extract_emg_data()`:
- **Differential mode** (optional): `diff[i] = emg[i+16] - emg[i]`, reduces 32→16 channels. Toggled in UI, updates `config.num_channels` and `config.active_channels` dynamically.
- **RealTimeEMGFilter**: Stateful IIR filter — bandpass 20-500 Hz + notch 50 Hz with 8 harmonics. State preserved between packets (continuous filtering via `rtfilter`).
- Filter has ~100ms warmup. Enable filter BEFORE starting recording/prediction.
- Filtering only happens in MuoviWidget — NOT in model.py (no double filtering).

### Configuration

Central config singleton in `mindmove/config.py` (instantiated as `config = config()`). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FSAMP` | 2000 | Sampling frequency (Hz) |
| `num_channels` | 32 | EMG channels (16 in differential mode) |
| `dead_channels` | [] | 0-indexed channels to exclude |
| `ENABLE_FILTERING` | True | Default filter state |
| `ENABLE_DIFFERENTIAL_MODE` | False | Monopolar (32ch) vs single differential (16ch) |
| `USE_NUMBA_DTW` | True | DTW implementation selection |
| `increment_dtw` | 0.05 | DTW interval in seconds (50ms) |
| `template_duration` | 1 | Template length in seconds |
| `window_length` | 192 | Feature window (~96ms at 2kHz) |
| `increment` | 64 | Feature window step (~32ms) |
| `POST_PREDICTION_SMOOTHING` | "MAJORITY VOTE" | Options: "NONE", "MAJORITY VOTE", "5 CONSECUTIVE" |
| `SMOOTHING_WINDOW` | 5 | Window for smoothing |
| `MIN_ACTIVATION_DURATION_S` | 1.5 | Min activation to extract template |
| `TARGET_TEMPLATES_PER_CLASS` | 20 | Target templates per class |
| `KINEMATICS_FSAMP` | 60 | VHI kinematics sampling rate |

### Data Directories

```
data/
├── templates_open/      # OPEN state templates
├── templates_closed/    # CLOSED state templates
├── recordings/          # Protocol outputs
├── datasets/            # Training datasets
├── models/              # Trained classifier models (.pkl)
└── predictions/         # Online session logs
```

## Key Patterns

- **Qt Signal/Slot** for all inter-component communication
- **Stateful filtering** — `RealTimeEMGFilter` preserves state between packets
- **QThread for training** — heavy computation in separate thread
- **Config singleton** — global `config` instance, mutated by `set_differential_mode()`
- **Feature registry** — dictionary-based feature selection in `features_registry.py`
- **Model facade** — always use `MindMoveInterface`, not `Model` directly
- **EMG data format** — always `(n_channels, n_samples)` i.e. `(32, N)` or `(16, N)` in differential mode
- **Dead channels** — 0-indexed internally, 1-indexed in UI (UI converts on input)
- **Spatial profiles** — L2-normalized for computation, max-normalized for plotting. Second half of 1s window only (matches runtime behavior). Stored in model .pkl as `spatial_profiles.open/closed` containing `per_template_rms`, `ref_profile`, `weights`, `consistency`
- **Template metadata** — stable 1-based IDs survive deletions. Stored as `metadata_open`/`metadata_closed` in combined pkl files
- **Per-class active channels** — DTW uses different channel subsets for OPEN vs CLOSED distance computation
- **Default threshold** — mid-gap: `(mean_intra + std_intra + mean_cross - std_cross) / 2`

## Common Issues

1. **Filter transients**: Wait 1-2 seconds after enabling filter before recording
2. **No double filtering**: Filtering only in MuoviWidget, never in model.py
3. **Differential mode side effects**: Changing mode updates `config.num_channels`, `config.active_channels`, and reinitializes the filter — downstream components must handle channel count changes
4. **Terminal output**: State transitions print immediately; status every 2s; timing summary every 5s
5. **Windowing**: `sliding_window()` 3rd param is SHIFT not overlap. `config.increment` (=64) is shift; don't confuse with overlap
6. **Spatial profiles use second half only**: Both `compute_spatial_profiles()` (training) and `compute_live_rms()` (runtime) use `emg[:, half:]` — the active portion of the 1s window
7. **NaN safety in per-template coupling**: Empty template arrays → `np.inf`, zero-norm RMS → `np.zeros`, NaN dots → 0.0, NaN corrected → `np.inf` before aggregation
8. **Online settings are live**: All online protocol UI changes (spatial mode, coupling, thresholds, decision model) take effect immediately — no restart needed. Both UI handlers and `online_emg_update()` run in main Qt thread sequentially
