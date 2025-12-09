# Seminar MindMove: Real-Time Decoding of Neural Signals into Movement Intent

MindMove is a Python-based framework designed for acquiring, processing, visualizing, and decoding neural signals, particularly Electromyography (EMG) data, in real-time. It provides a graphical user interface (GUI) built with PySide6 and leverages VisPy for efficient real-time data plotting. The framework is intended to interface with specific hardware (e.g., the "Muovi" device) and potentially control external systems like virtual hands based on the decoded movement intent.

## Core Functionality

*   **Data Acquisition:** Interfaces with hardware (e.g., via `MuoviWidget`) to stream real-time sensor data.
*   **Signal Processing:** Extracts relevant features from the raw data (e.g., EMG envelopes).
*   **Real-time Visualization:** Plots multi-channel signal data dynamically using VisPy, allowing for immediate feedback and monitoring.
*   **Decoding/Control Interface:** Includes components like `VirtualHandInterface` suggesting capabilities for translating processed signals into control commands (e.g., for a virtual prosthesis).
*   **Model Integration:** Provides a structure (`mindmove/model`) for integrating machine learning models for signal classification or regression tasks.

## Project Structure

```
├─── .git/                  # Git version control files
├─── .idea/                 # IDE configuration (e.g., PyCharm)
├─── .venv/                 # Virtual environment (if created locally)
├─── mindmove/              # Main application source code
│   ├─── __init__.py
│   ├─── device_interfaces/ # Code for interfacing with hardware devices (e.g., MuoviWidget)
│   ├─── gui/               # Core GUI components (main window, tabs, etc.)
│   │   ├─── ui_compiled/   # Compiled UI files from Qt Designer (.ui -> .py)
│   │   ├─── __init__.py
│   │   ├─── mindmove.py    # Main application window class
│   │   ├─── protocol.py    # Handles specific protocols or procedures
│   │   └─── virtual_hand_interface.py # Interface for controlling a virtual hand
│   ├─── gui_custom_elements/ # Custom Qt widgets (e.g., VispyPlotWidget)
│   ├─── model/             # Machine learning models and related logic
│   │   ├─── core/
│   │   |   ├─── datasets.py # Implementation for custom dataset handling
│   │   |   └─── model.py    # Implementation for ML models
│   │   ├─── tests/          # Tests for the model components
│   │   └─── interface.py    # Functions connecting the model logic to the GUI
│   └─── main.py            # Main application entry point
├─── data/                  # Local data storage (e.g., recordings, model weights - *Do not commit to Git*)
├─── test/                  # General project tests (if any)
├─── .gitignore             # Specifies intentionally untracked files that Git should ignore
├─── .python-version        # Specifies the Python version (used by pyenv)
├─── hello.py               # Simple test script (likely removable)
├─── poetry.lock            # Exact dependency versions locked by Poetry
├─── poetry.toml            # Poetry configuration
├─── pyproject.toml         # Project metadata and dependencies for Poetry
├─── README.md              # This file
└─── uv.lock                # uv dependency lock file (alternative/complement to poetry.lock)
```

## Getting Started

### Prerequisites

*   **Python:** Version 3.11 ([Download](https://www.python.org/downloads/release/python-3119/))
*   **IDE:** An Integrated Development Environment is recommended (e.g., [PyCharm](https://www.jetbrains.com/pycharm/download/), [VS Code](https://code.visualstudio.com/download))
*   **UV:** An extremely fast Python package installer and resolver ([Installation Guide](https://github.com/astral-sh/uv#installation))

### Installation

1.  **(Optional)** If you need to use SSH for cloning, add your SSH key to your `gitos.rrze.fau.de` account ([Tutorial](https://docs.gitlab.com/ee/user/ssh.html)).
    ```bash
    # Generate SSH key if you don't have one
    ssh-keygen -t ed25519 -C "your_email@example.com"
    # Follow prompts and then add the public key (~/.ssh/id_ed25519.pub) to GitLab
    ```

2.  **Clone the repository:** Choose either HTTPS or SSH. Replace `git-url` with the actual repository URL.
    ```bash
    # Using HTTPS (replace with actual URL)
    # git clone https://gitos.rrze.fau.de/path/to/repository.git

    # Or using SSH (replace with actual URL)
    # git clone git@gitos.rrze.fau.de:path/to/repository.git

    # cd mindmove-framework # Or your repository directory name
    ```
    *Note: Make sure to replace the placeholder URLs above with the correct ones for this project.*

3.  **Install dependencies:** Navigate to the project root directory in your terminal and use UV to install the required packages from the lock file (`uv.lock` or `poetry.lock`) or `pyproject.toml`. UV will create and manage a virtual environment.
    ```bash
    uv sync
    ```
    *Note: This command reads the project configuration and installs the exact dependencies specified in the lock file, ensuring reproducible environments.*

4.  **Add new dependencies:** To add a new package to the project:
    ```bash
    uv pip install <package_name>
    ```
    *Note: This command installs the package and updates the `pyproject.toml` and lock file.*

5.  **(Optional) Install PyTorch:** If your models require PyTorch, install it separately using `uv pip install`. Adjust the command based on your system (CPU/GPU - CUDA version). The example below uses CUDA 12.1.
    ```bash
    uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    *Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command for your setup.*

### Running the Application

Once the installation is complete, you can run the main MindMove application using `uv run`:

```bash
uv run python mindmove/main.py
```
*The `uv run` command ensures the script executes within the virtual environment managed by UV.*

## Contact

*   Dominik Braun - dome.braun@fau.de
*   Raul Sîmpetru - raul.simpetru@fau.de

Original Repository Link: [https://gitos.rrze.fau.de/n-squared-lab/teaching/ss-24/mindmove/mindmove-framework](https://gitos.rrze.fau.de/n-squared-lab/teaching/ss-24/mindmove/mindmove-framework)




