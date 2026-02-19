"""
Lightweight decision models for patient-specific state classification.

Replaces the manual threshold + spatial correction logic with ML models
trained on the patient's calibration data (guided recording + model).

Supported models:
1. Neural Network: Dense(n,16)->ReLU->Dense(16,8)->ReLU->Dense(8,1)->Sigmoid
2. CatBoost (Gradient Boosting): Tree ensemble with derived features

Input:  [D_open, D_closed, sim_open, sim_closed]  (4 features with spatial)
        or [D_open, D_closed]                      (2 features without spatial)
Output: P(CLOSED) in [0, 1]

Training: ~600 labeled samples from 10-cycle guided recording, ~2s training time.
"""

import numpy as np


class DecisionNetworkInference:
    """Pure numpy inference for the decision network. No PyTorch required.

    Supports both 3-layer (recording-based) and 2-layer (template-based) architectures.
    """

    def __init__(self, weights: dict):
        """
        Initialize from a weights dictionary (extracted from PyTorch after training).

        Args:
            weights: dict with keys W1, b1, W2, b2, [W3, b3], input_mean, input_std
        """
        self.W1 = np.array(weights['W1'], dtype=np.float32)
        self.b1 = np.array(weights['b1'], dtype=np.float32)
        self.W2 = np.array(weights['W2'], dtype=np.float32)
        self.b2 = np.array(weights['b2'], dtype=np.float32)
        # 3-layer architecture (recording-based): W3/b3 present
        # 2-layer architecture (template-based): W3/b3 absent
        self.W3 = np.array(weights['W3'], dtype=np.float32) if 'W3' in weights else None
        self.b3 = np.array(weights['b3'], dtype=np.float32) if 'b3' in weights else None
        self.n_layers = 3 if self.W3 is not None else 2
        self.input_mean = np.array(weights['input_mean'], dtype=np.float32)
        self.input_std = np.array(weights['input_std'], dtype=np.float32)
        self.n_inputs = int(weights.get('n_inputs', len(self.input_mean)))
        self.has_spatial = bool(weights.get('has_spatial', self.n_inputs == 4))
        self.accuracy = float(weights.get('accuracy', 0.0))

    def predict(self, features: np.ndarray) -> float:
        """
        Predict P(CLOSED) from input features.

        Args:
            features: array of shape (n_inputs,)
                      [D_open, D_closed, sim_open, sim_closed] or [D_open, D_closed]

        Returns:
            float: P(CLOSED) in [0, 1]
        """
        # Normalize
        x = (features - self.input_mean) / (self.input_std + 1e-8)

        # Layer 1: ReLU
        h = x @ self.W1.T + self.b1
        h = np.maximum(h, 0)

        if self.n_layers == 3:
            # Layer 2: ReLU
            h = h @ self.W2.T + self.b2
            h = np.maximum(h, 0)
            # Output: Sigmoid
            logit = float(h @ self.W3.T + self.b3)
        else:
            # Output: Sigmoid (W2 is the output layer)
            logit = float(h @ self.W2.T + self.b2)

        logit = np.clip(logit, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-logit))

        return float(prob)

    def get_weights_dict(self) -> dict:
        """Export weights as a plain dict (for pickle serialization)."""
        d = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'n_inputs': self.n_inputs,
            'has_spatial': self.has_spatial,
            'accuracy': self.accuracy,
        }
        if self.W3 is not None:
            d['W3'] = self.W3
            d['b3'] = self.b3
        return d


def _collect_training_features(
    emg_data: np.ndarray,
    gt_data: np.ndarray,
    templates_open: list,
    templates_closed: list,
    feature_name: str,
    distance_aggregation: str = 'average',
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Run offline DTW and collect features + labels for NN training.

    Returns:
        (X, y, has_spatial, timestamps) where:
            X: (n_ticks, n_features) feature matrix
            y: (n_ticks,) binary labels (0=OPEN, 1=CLOSED)
            has_spatial: bool
            timestamps: (n_ticks,) time in seconds
    """
    from mindmove.model.offline_test import simulate_realtime_dtw
    from mindmove.config import config

    has_spatial = (spatial_ref_open is not None and spatial_ref_closed is not None)

    # Run offline simulation to get distances (and spatial sims if available)
    # Use spatial_mode="scaling" just to trigger similarity computation
    spatial_mode = "scaling" if has_spatial else "off"

    results = simulate_realtime_dtw(
        emg_data=emg_data,
        templates_open=templates_open,
        templates_closed=templates_closed,
        threshold_open=1.0,   # Doesn't matter, we only want features
        threshold_closed=1.0,
        feature_name=feature_name,
        verbose=False,
        distance_aggregation=distance_aggregation,
        spatial_ref_open=spatial_ref_open,
        spatial_ref_closed=spatial_ref_closed,
        spatial_threshold=0.5,
        spatial_mode=spatial_mode,
        initial_state="CLOSED",
    )

    timestamps = results['timestamps']
    D_open = np.array(results['D_open'])
    D_closed = np.array(results['D_closed'])
    n_ticks = len(timestamps)

    # Build feature matrix
    if has_spatial:
        sim_open = np.array([s if s is not None else 0.0 for s in results['sim_open']])
        sim_closed = np.array([s if s is not None else 0.0 for s in results['sim_closed']])
        X = np.column_stack([D_open, D_closed, sim_open, sim_closed])
    else:
        X = np.column_stack([D_open, D_closed])

    # Interpolate GT to DTW timestamps
    gt = np.array(gt_data, dtype=float).flatten()
    gt_time = np.arange(len(gt)) / config.FSAMP
    y = np.interp(timestamps, gt_time, gt)
    y = (y > 0.5).astype(np.float32)  # Binarize: 0=OPEN, 1=CLOSED

    if verbose:
        n_closed = int(y.sum())
        n_open = n_ticks - n_closed
        feature_names = ['D_open', 'D_closed', 'sim_open', 'sim_closed'] if has_spatial else ['D_open', 'D_closed']
        print(f"\n[DECISION NN] Training data:")
        print(f"  Ticks: {n_ticks}")
        print(f"  Features: {len(feature_names)} ({', '.join(feature_names)})")
        print(f"  Labels: {n_closed} CLOSED, {n_open} OPEN")
        for i, name in enumerate(feature_names):
            print(f"    {name}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
                  f"range=[{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    return X, y, has_spatial, timestamps


def _collect_training_features_from_templates(
    templates_open_features: list,
    templates_closed_features: list,
    distance_aggregation: str = 'average',
    templates_open_raw: list = None,
    templates_closed_raw: list = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Collect training features from templates only (no recording needed).

    For each template, computes DTW distance to both classes (leave-one-out
    for own class) and optional spatial similarity. Labels are perfect since
    we know exactly which class each template belongs to.

    Args:
        templates_open_features: Feature-extracted OPEN templates
        templates_closed_features: Feature-extracted CLOSED templates
        distance_aggregation: DTW distance aggregation method
        templates_open_raw: Raw EMG OPEN templates (for spatial similarity)
        templates_closed_raw: Raw EMG CLOSED templates (for spatial similarity)
        spatial_ref_open: Spatial profile for OPEN class
        spatial_ref_closed: Spatial profile for CLOSED class

    Returns:
        (X, y, has_spatial) where:
            X: (n_templates, n_features) feature matrix
            y: (n_templates,) binary labels (0=OPEN, 1=CLOSED)
            has_spatial: bool
    """
    from mindmove.model.core.algorithm import (
        compute_distance_from_training_set_online,
        compute_spatial_similarity,
    )

    has_spatial = (spatial_ref_open is not None and spatial_ref_closed is not None
                   and templates_open_raw is not None and templates_closed_raw is not None)

    n_open = len(templates_open_features)
    n_closed = len(templates_closed_features)

    features_list = []
    labels = []

    # Process OPEN templates
    for i, tpl in enumerate(templates_open_features):
        # Leave-one-out: exclude self from open templates
        other_open = [t for j, t in enumerate(templates_open_features) if j != i]

        D_open = compute_distance_from_training_set_online(
            tpl, other_open, distance_aggregation=distance_aggregation
        )
        D_closed = compute_distance_from_training_set_online(
            tpl, templates_closed_features, distance_aggregation=distance_aggregation
        )

        row = [D_open, D_closed]

        if has_spatial:
            sim_open = compute_spatial_similarity(
                templates_open_raw[i],
                spatial_ref_open["ref_profile"], spatial_ref_open["weights"]
            )
            sim_closed = compute_spatial_similarity(
                templates_open_raw[i],
                spatial_ref_closed["ref_profile"], spatial_ref_closed["weights"]
            )
            row.extend([sim_open, sim_closed])

        features_list.append(row)
        labels.append(0)  # OPEN

    # Process CLOSED templates
    for i, tpl in enumerate(templates_closed_features):
        other_closed = [t for j, t in enumerate(templates_closed_features) if j != i]

        D_open = compute_distance_from_training_set_online(
            tpl, templates_open_features, distance_aggregation=distance_aggregation
        )
        D_closed = compute_distance_from_training_set_online(
            tpl, other_closed, distance_aggregation=distance_aggregation
        )

        row = [D_open, D_closed]

        if has_spatial:
            sim_open = compute_spatial_similarity(
                templates_closed_raw[i],
                spatial_ref_open["ref_profile"], spatial_ref_open["weights"]
            )
            sim_closed = compute_spatial_similarity(
                templates_closed_raw[i],
                spatial_ref_closed["ref_profile"], spatial_ref_closed["weights"]
            )
            row.extend([sim_open, sim_closed])

        features_list.append(row)
        labels.append(1)  # CLOSED

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    if verbose:
        feature_names = ['D_open', 'D_closed']
        if has_spatial:
            feature_names += ['sim_open', 'sim_closed']
        print(f"\n[DECISION] Template-based training data:")
        print(f"  Samples: {n_open + n_closed} ({n_open} OPEN, {n_closed} CLOSED)")
        print(f"  Features: {len(feature_names)} ({', '.join(feature_names)})")
        for i, name in enumerate(feature_names):
            print(f"    {name}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
                  f"range=[{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    return X, y, has_spatial


def train_decision_network(
    emg_data: np.ndarray,
    gt_data: np.ndarray,
    templates_open: list,
    templates_closed: list,
    feature_name: str,
    distance_aggregation: str = 'average',
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    epochs: int = 300,
    lr: float = 0.005,
    verbose: bool = True,
) -> dict:
    """
    Train the decision network on a guided recording.

    Args:
        emg_data: EMG data (n_channels, n_samples)
        gt_data: Ground truth array at FSAMP rate (0=OPEN, 1=CLOSED)
        templates_open: Feature templates for OPEN class
        templates_closed: Feature templates for CLOSED class
        feature_name: Feature extraction method name
        distance_aggregation: DTW distance aggregation method
        spatial_ref_open: Spatial profile for OPEN (or None)
        spatial_ref_closed: Spatial profile for CLOSED (or None)
        epochs: Training epochs
        lr: Learning rate
        verbose: Print progress

    Returns:
        dict with NN weights and metadata (ready for DecisionNetworkInference)
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError(
            "PyTorch is required for training the decision network.\n"
            "Install with: uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    # Collect features
    X, y, has_spatial, timestamps = _collect_training_features(
        emg_data, gt_data, templates_open, templates_closed,
        feature_name, distance_aggregation,
        spatial_ref_open, spatial_ref_closed, verbose,
    )

    n_ticks = len(y)
    n_inputs = X.shape[1]

    # Normalize features
    input_mean = X.mean(axis=0).astype(np.float32)
    input_std = X.std(axis=0).astype(np.float32)
    X_norm = ((X - input_mean) / (input_std + 1e-8)).astype(np.float32)

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    # Define model
    model = nn.Sequential(
        nn.Linear(n_inputs, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    best_accuracy = 0.0
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        # Track best model
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = (model(X_tensor) > 0.5).float()
                accuracy = (preds == y_tensor).float().mean().item()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                preds = (model(X_tensor) > 0.5).float()
                accuracy = (preds == y_tensor).float().mean().item()
                print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, accuracy={accuracy:.1%}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()
        preds = (probs > 0.5).astype(float)
        accuracy = float(np.mean(preds == y))

        # Per-class accuracy
        open_mask = y == 0
        closed_mask = y == 1
        acc_open = float(np.mean(preds[open_mask] == 0)) if open_mask.any() else 0.0
        acc_closed = float(np.mean(preds[closed_mask] == 1)) if closed_mask.any() else 0.0

    n_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"\n[DECISION NN] Training complete:")
        print(f"  Overall accuracy: {accuracy:.1%}")
        print(f"  OPEN accuracy:    {acc_open:.1%}")
        print(f"  CLOSED accuracy:  {acc_closed:.1%}")
        print(f"  Parameters:       {n_params}")

    # Extract weights as numpy arrays
    params = list(model.parameters())
    weights = {
        'W1': params[0].detach().numpy(),   # (16, n_inputs)
        'b1': params[1].detach().numpy(),   # (16,)
        'W2': params[2].detach().numpy(),   # (8, 16)
        'b2': params[3].detach().numpy(),   # (8,)
        'W3': params[4].detach().numpy(),   # (1, 8)
        'b3': params[5].detach().numpy(),   # (1,)
        'input_mean': input_mean,
        'input_std': input_std,
        'n_inputs': n_inputs,
        'has_spatial': has_spatial,
        'accuracy': accuracy,
        'accuracy_open': acc_open,
        'accuracy_closed': acc_closed,
        'n_training_samples': n_ticks,
        'n_params': n_params,
    }

    return weights


# ============================================================
# Template-based training (no recording needed)
# ============================================================

def train_decision_network_from_templates(
    templates_open_features: list,
    templates_closed_features: list,
    distance_aggregation: str = 'average',
    templates_open_raw: list = None,
    templates_closed_raw: list = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    epochs: int = 500,
    lr: float = 0.01,
    verbose: bool = True,
) -> dict:
    """Train NN decision model using only template-vs-template distances."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        raise ImportError(
            "PyTorch is required for training the decision network.\n"
            "Install with: uv pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )

    X, y, has_spatial = _collect_training_features_from_templates(
        templates_open_features, templates_closed_features,
        distance_aggregation,
        templates_open_raw, templates_closed_raw,
        spatial_ref_open, spatial_ref_closed, verbose,
    )

    n_samples = len(y)
    n_inputs = X.shape[1]

    # Normalize
    input_mean = X.mean(axis=0).astype(np.float32)
    input_std = X.std(axis=0).astype(np.float32)
    X_norm = ((X - input_mean) / (input_std + 1e-8)).astype(np.float32)

    X_tensor = torch.from_numpy(X_norm)
    y_tensor = torch.from_numpy(y).unsqueeze(1)

    # Smaller network for smaller dataset
    model = nn.Sequential(
        nn.Linear(n_inputs, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    best_accuracy = 0.0
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = (model(X_tensor) > 0.5).float()
                accuracy = (preds == y_tensor).float().mean().item()
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 100 == 0:
            with torch.no_grad():
                preds = (model(X_tensor) > 0.5).float()
                accuracy = (preds == y_tensor).float().mean().item()
                print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, accuracy={accuracy:.1%}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()
        preds = (probs > 0.5).astype(float)
        accuracy = float(np.mean(preds == y))

        open_mask = y == 0
        closed_mask = y == 1
        acc_open = float(np.mean(preds[open_mask] == 0)) if open_mask.any() else 0.0
        acc_closed = float(np.mean(preds[closed_mask] == 1)) if closed_mask.any() else 0.0

    n_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"\n[DECISION NN] Training complete (template-based):")
        print(f"  Overall accuracy: {accuracy:.1%}")
        print(f"  OPEN accuracy:    {acc_open:.1%}")
        print(f"  CLOSED accuracy:  {acc_closed:.1%}")
        print(f"  Parameters:       {n_params}")

    params = list(model.parameters())
    weights = {
        'W1': params[0].detach().numpy(),
        'b1': params[1].detach().numpy(),
        'W2': params[2].detach().numpy(),
        'b2': params[3].detach().numpy(),
        'input_mean': input_mean,
        'input_std': input_std,
        'n_inputs': n_inputs,
        'has_spatial': has_spatial,
        'accuracy': accuracy,
        'accuracy_open': acc_open,
        'accuracy_closed': acc_closed,
        'n_training_samples': n_samples,
        'n_params': n_params,
    }

    return weights


def train_decision_catboost_from_templates(
    templates_open_features: list,
    templates_closed_features: list,
    distance_aggregation: str = 'average',
    templates_open_raw: list = None,
    templates_closed_raw: list = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    iterations: int = 200,
    depth: int = 3,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> dict:
    """Train CatBoost decision model using only template-vs-template distances."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError(
            "CatBoost is required for training the gradient boosting model.\n"
            "Install with: uv pip install catboost"
        )

    X_raw, y, has_spatial = _collect_training_features_from_templates(
        templates_open_features, templates_closed_features,
        distance_aggregation,
        templates_open_raw, templates_closed_raw,
        spatial_ref_open, spatial_ref_closed, verbose,
    )

    n_samples = len(y)
    n_raw_inputs = X_raw.shape[1]

    # Add derived features
    X = _add_derived_features(X_raw, has_spatial)
    feature_names = _get_feature_names(has_spatial)

    if verbose:
        print(f"\n[CATBOOST] Extended features: {X.shape[1]} ({', '.join(feature_names)})")
        for i, name in enumerate(feature_names):
            if i >= n_raw_inputs:
                print(f"    {name}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
                      f"range=[{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function='Logloss',
        verbose=False,
        random_seed=42,
        auto_class_weights='Balanced',
    )

    if verbose:
        print(f"[CATBOOST] Training with {iterations} iterations, depth={depth} (template-based, {n_samples} samples)...")

    from catboost import Pool
    train_pool = Pool(X, y.astype(int), feature_names=feature_names)
    model.fit(train_pool)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(float)
    accuracy = float(np.mean(preds == y))

    open_mask = y == 0
    closed_mask = y == 1
    acc_open = float(np.mean(preds[open_mask] == 0)) if open_mask.any() else 0.0
    acc_closed = float(np.mean(preds[closed_mask] == 1)) if closed_mask.any() else 0.0

    n_trees = model.tree_count_

    if verbose:
        print(f"\n[CATBOOST] Training complete (template-based):")
        print(f"  Overall accuracy: {accuracy:.1%}")
        print(f"  OPEN accuracy:    {acc_open:.1%}")
        print(f"  CLOSED accuracy:  {acc_closed:.1%}")
        print(f"  Trees:            {n_trees}")

        importances = model.get_feature_importance()
        print(f"  Feature importance:")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            print(f"    {feature_names[idx]:<15s} {importances[idx]:6.1f}%")

    import tempfile
    import os

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cbm')
    os.close(tmp_fd)
    try:
        model.save_model(tmp_path)
        with open(tmp_path, 'rb') as f:
            model_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    model_data = {
        'model_bytes': model_bytes,
        'n_inputs': n_raw_inputs,
        'has_spatial': has_spatial,
        'accuracy': accuracy,
        'accuracy_open': acc_open,
        'accuracy_closed': acc_closed,
        'n_training_samples': n_samples,
        'n_trees': n_trees,
        'feature_names': feature_names,
    }

    return model_data


# ============================================================
# CatBoost (Gradient Boosting) Decision Model
# ============================================================

def _add_derived_features(X: np.ndarray, has_spatial: bool) -> np.ndarray:
    """
    Add derived features that help tree-based models capture ratios/differences.

    Raw features: [D_open, D_closed] or [D_open, D_closed, sim_open, sim_closed]
    Added:
        D_ratio     = D_open / (D_closed + 1e-8)
        D_diff      = D_open - D_closed
        sim_contrast = sim_closed - sim_open  (only if spatial)

    Returns: extended feature matrix
    """
    D_open = X[:, 0]
    D_closed = X[:, 1]

    D_ratio = D_open / (D_closed + 1e-8)
    D_diff = D_open - D_closed

    derived = [D_ratio.reshape(-1, 1), D_diff.reshape(-1, 1)]

    if has_spatial and X.shape[1] >= 4:
        sim_open = X[:, 2]
        sim_closed = X[:, 3]
        sim_contrast = (sim_closed - sim_open).reshape(-1, 1)
        derived.append(sim_contrast)

    return np.hstack([X] + derived)


def _get_feature_names(has_spatial: bool) -> list:
    """Get feature names for the extended feature set."""
    names = ['D_open', 'D_closed']
    if has_spatial:
        names += ['sim_open', 'sim_closed']
    names += ['D_ratio', 'D_diff']
    if has_spatial:
        names += ['sim_contrast']
    return names


class CatBoostDecisionInference:
    """CatBoost inference for the gradient boosting decision model."""

    def __init__(self, model_data: dict):
        """
        Initialize from a model data dictionary.

        Args:
            model_data: dict with keys model_bytes, n_inputs, has_spatial, accuracy, feature_names
        """
        self.n_inputs = int(model_data['n_inputs'])  # raw input count
        self.has_spatial = bool(model_data['has_spatial'])
        self.accuracy = float(model_data.get('accuracy', 0.0))
        self.feature_names = model_data.get('feature_names', [])
        self.n_trees = int(model_data.get('n_trees', 0))
        # True → transition proximity detector (state-conditioned features at inference)
        # False → legacy posture classifier (global [D_open, D_closed, sim_open, sim_closed])
        self.transition_mode = bool(model_data.get('transition_mode', False))

        # Load CatBoost model from bytes
        from catboost import CatBoostClassifier
        import tempfile
        import os

        self._model = CatBoostClassifier()
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cbm')
        try:
            with os.fdopen(tmp_fd, 'wb') as f:
                f.write(model_data['model_bytes'])
            self._model.load_model(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _extend_features(self, features: np.ndarray) -> np.ndarray:
        """Add derived features to raw input (same transforms as training)."""
        D_open, D_closed = features[0], features[1]
        derived = [D_open / (D_closed + 1e-8), D_open - D_closed]
        if self.has_spatial and len(features) >= 4:
            derived.append(features[3] - features[2])  # sim_closed - sim_open
        return np.concatenate([features, derived])

    def predict(self, features: np.ndarray) -> float:
        """
        Predict P(CLOSED) from raw input features (posture-classifier / legacy mode).

        Args:
            features: array of shape (n_inputs,)
                      [D_open, D_closed, sim_open, sim_closed] or [D_open, D_closed]

        Returns:
            float: P(CLOSED) in [0, 1]
        """
        extended = self._extend_features(features)
        probs = self._model.predict_proba(extended.reshape(1, -1))
        return float(probs[0, 1])  # P(class=1) = P(CLOSED)

    def predict_transition(self, D_target: float, sim_target: float = None) -> float:
        """
        Predict P(matches target class) using state-conditioned features (transition mode).

        This is called with features from the TARGET class only — the class we are
        checking whether to transition to. The caller chooses D_target / sim_target
        based on the current state:
            state=OPEN,   checking CLOSED: D_target=D_closed, sim_target=sim_closed
            state=CLOSED, checking OPEN:   D_target=D_open,   sim_target=sim_open

        Args:
            D_target:   DTW distance to target class templates
            sim_target: Spatial similarity to target class profile (None if not available)

        Returns:
            float: P(transition) in [0, 1] — probability that current window matches target class
        """
        row = [D_target]
        if self.has_spatial and sim_target is not None:
            row.append(sim_target)
        features = np.array(row, dtype=np.float32)
        probs = self._model.predict_proba(features.reshape(1, -1))
        return float(probs[0, 1])

    def get_model_dict(self) -> dict:
        """Export model data as a plain dict (for pickle serialization)."""
        import tempfile
        import os

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cbm')
        os.close(tmp_fd)
        try:
            self._model.save_model(tmp_path)
            with open(tmp_path, 'rb') as f:
                model_bytes = f.read()
        finally:
            os.unlink(tmp_path)

        return {
            'model_bytes': model_bytes,
            'n_inputs': self.n_inputs,
            'has_spatial': self.has_spatial,
            'accuracy': self.accuracy,
            'feature_names': self.feature_names,
            'n_trees': self.n_trees,
            'transition_mode': self.transition_mode,
        }


def _collect_transition_features_from_templates(
    templates_open_features: list,
    templates_closed_features: list,
    distance_aggregation: str = 'average',
    templates_open_raw: list = None,
    templates_closed_raw: list = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Collect training features for a TRANSITION PROXIMITY DETECTOR.

    Instead of classifying posture globally (which class does this look like?),
    this trains a model that answers: "does this window look like a TARGET class template?"

    Features (always from target-class perspective):
        D_target    — DTW distance to target class templates (small = good temporal match)
        sim_target  — Spatial similarity to target class profile (high = right muscles, optional)

    Training data is symmetric — both transition directions are included:
      - OPEN  templates: positive for "OPEN target"  + negative for "CLOSED target"
      - CLOSED templates: positive for "CLOSED target" + negative for "OPEN target"

    Returns: (X, y, has_spatial, feature_names)
        X: (2*(n_open+n_closed), n_features)
        y: (2*(n_open+n_closed),) — 1 = matches target class, 0 = does not
    """
    from mindmove.model.core.algorithm import (
        compute_distance_from_training_set_online,
        compute_spatial_similarity,
    )

    has_spatial = (spatial_ref_open is not None and spatial_ref_closed is not None
                   and templates_open_raw is not None and templates_closed_raw is not None)

    n_open = len(templates_open_features)
    n_closed = len(templates_closed_features)

    features_list = []
    labels = []

    def _make_row(D_t, sim_t):
        row = [D_t]
        if has_spatial and sim_t is not None:
            row.append(sim_t)
        return row

    # ── OPEN as target class ─────────────────────────────────────────
    # Positive: each OPEN template vs remaining OPEN templates (leave-one-out)
    for i, tpl in enumerate(templates_open_features):
        other_open = [t for j, t in enumerate(templates_open_features) if j != i]
        D_t = compute_distance_from_training_set_online(tpl, other_open, distance_aggregation=distance_aggregation)
        sim_t = (compute_spatial_similarity(
                     templates_open_raw[i], spatial_ref_open["ref_profile"], spatial_ref_open["weights"])
                 if has_spatial else None)
        features_list.append(_make_row(D_t, sim_t))
        labels.append(1)  # matches OPEN target → should trigger OPEN

    # Negative: each CLOSED template checked against all OPEN templates
    for i, tpl in enumerate(templates_closed_features):
        D_t = compute_distance_from_training_set_online(tpl, templates_open_features, distance_aggregation=distance_aggregation)
        sim_t = (compute_spatial_similarity(
                     templates_closed_raw[i], spatial_ref_open["ref_profile"], spatial_ref_open["weights"])
                 if has_spatial else None)
        features_list.append(_make_row(D_t, sim_t))
        labels.append(0)  # CLOSED does not match OPEN target → should not trigger

    # ── CLOSED as target class ────────────────────────────────────────
    # Positive: each CLOSED template vs remaining CLOSED templates (leave-one-out)
    for i, tpl in enumerate(templates_closed_features):
        other_closed = [t for j, t in enumerate(templates_closed_features) if j != i]
        D_t = compute_distance_from_training_set_online(tpl, other_closed, distance_aggregation=distance_aggregation)
        sim_t = (compute_spatial_similarity(
                     templates_closed_raw[i], spatial_ref_closed["ref_profile"], spatial_ref_closed["weights"])
                 if has_spatial else None)
        features_list.append(_make_row(D_t, sim_t))
        labels.append(1)  # matches CLOSED target → should trigger CLOSED

    # Negative: each OPEN template checked against all CLOSED templates
    for i, tpl in enumerate(templates_open_features):
        D_t = compute_distance_from_training_set_online(tpl, templates_closed_features, distance_aggregation=distance_aggregation)
        sim_t = (compute_spatial_similarity(
                     templates_open_raw[i], spatial_ref_closed["ref_profile"], spatial_ref_closed["weights"])
                 if has_spatial else None)
        features_list.append(_make_row(D_t, sim_t))
        labels.append(0)  # OPEN does not match CLOSED target → should not trigger

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    feature_names = ['D_target']
    if has_spatial:
        feature_names += ['sim_target']

    if verbose:
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        print(f"\n[CATBOOST TRANSITION] Training data:")
        print(f"  Samples: {len(y)} ({n_pos} positive / {n_neg} negative)")
        print(f"  From: {n_open} OPEN + {n_closed} CLOSED templates")
        print(f"  Features: {len(feature_names)} ({', '.join(feature_names)})")
        for i, name in enumerate(feature_names):
            vals = X[:, i]
            print(f"    {name}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                  f"range=[{vals.min():.4f}, {vals.max():.4f}]")
        # Separability hint
        pos_D = X[y == 1, 0]
        neg_D = X[y == 0, 0]
        print(f"  Separability — D_target: pos mean={pos_D.mean():.4f}, neg mean={neg_D.mean():.4f}")

    return X, y, has_spatial, feature_names


def train_transition_catboost_from_templates(
    templates_open_features: list,
    templates_closed_features: list,
    distance_aggregation: str = 'average',
    templates_open_raw: list = None,
    templates_closed_raw: list = None,
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    iterations: int = 200,
    depth: int = 3,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Train CatBoost as a TRANSITION PROXIMITY DETECTOR (not a posture classifier).

    The model answers: "does the current window look like a TARGET class template?"
    Features are always [D_target, sim_target, D_over_sim] from the target-class perspective.

    At inference time (model.py / offline_test.py), features are state-conditioned:
        state=OPEN,   checking CLOSED: feed [D_closed, sim_closed, D_closed/sim_closed]
        state=CLOSED, checking OPEN:   feed [D_open,   sim_open,   D_open/sim_open]

    This eliminates false triggers from rising current-class distance during ADLs.
    The model is stored with 'transition_mode': True to signal this behavior.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("CatBoost required: uv pip install catboost")

    X, y, has_spatial, feature_names = _collect_transition_features_from_templates(
        templates_open_features, templates_closed_features,
        distance_aggregation,
        templates_open_raw, templates_closed_raw,
        spatial_ref_open, spatial_ref_closed, verbose,
    )

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function='Logloss',
        verbose=False,
        random_seed=42,
        auto_class_weights='Balanced',
    )

    if verbose:
        print(f"[CATBOOST TRANSITION] Training ({iterations} iter, depth={depth}, {len(y)} samples)...")

    train_pool = Pool(X, y.astype(int), feature_names=feature_names)
    model.fit(train_pool)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(float)
    accuracy = float(np.mean(preds == y))
    pos_mask = y == 1
    neg_mask = y == 0
    acc_pos = float(np.mean(preds[pos_mask] == 1)) if pos_mask.any() else 0.0
    acc_neg = float(np.mean(preds[neg_mask] == 0)) if neg_mask.any() else 0.0
    n_trees = model.tree_count_

    if verbose:
        print(f"\n[CATBOOST TRANSITION] Training complete:")
        print(f"  Overall accuracy:  {accuracy:.1%}")
        print(f"  Positive (match):  {acc_pos:.1%}")
        print(f"  Negative (no match): {acc_neg:.1%}")
        print(f"  Trees: {n_trees}")
        importances = model.get_feature_importance()
        sorted_idx = np.argsort(importances)[::-1]
        print(f"  Feature importance:")
        for idx in sorted_idx:
            print(f"    {feature_names[idx]:<15s} {importances[idx]:6.1f}%")

    import tempfile, os
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cbm')
    os.close(tmp_fd)
    try:
        model.save_model(tmp_path)
        with open(tmp_path, 'rb') as f:
            model_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    return {
        'model_bytes': model_bytes,
        'n_inputs': len(feature_names),
        'has_spatial': has_spatial,
        'transition_mode': True,  # state-conditioned inference
        'accuracy': accuracy,
        'accuracy_positive': acc_pos,
        'accuracy_negative': acc_neg,
        'n_training_samples': len(y),
        'n_trees': n_trees,
        'feature_names': feature_names,
    }


def train_decision_catboost(
    emg_data: np.ndarray,
    gt_data: np.ndarray,
    templates_open: list,
    templates_closed: list,
    feature_name: str,
    distance_aggregation: str = 'average',
    spatial_ref_open: dict = None,
    spatial_ref_closed: dict = None,
    iterations: int = 300,
    depth: int = 4,
    learning_rate: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Train a CatBoost gradient boosting model on a guided recording.

    Uses derived features (ratio, difference, contrast) in addition to raw
    DTW distances and spatial similarities — these help tree-based models
    capture relationships that would otherwise require many splits.

    Args:
        emg_data: EMG data (n_channels, n_samples)
        gt_data: Ground truth array at FSAMP rate (0=OPEN, 1=CLOSED)
        templates_open: Feature templates for OPEN class
        templates_closed: Feature templates for CLOSED class
        feature_name: Feature extraction method name
        distance_aggregation: DTW distance aggregation method
        spatial_ref_open: Spatial profile for OPEN (or None)
        spatial_ref_closed: Spatial profile for CLOSED (or None)
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        verbose: Print progress

    Returns:
        dict with CatBoost model bytes and metadata (ready for CatBoostDecisionInference)
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError(
            "CatBoost is required for training the gradient boosting model.\n"
            "Install with: uv pip install catboost"
        )

    # Collect raw features (same as NN)
    X_raw, y, has_spatial, timestamps = _collect_training_features(
        emg_data, gt_data, templates_open, templates_closed,
        feature_name, distance_aggregation,
        spatial_ref_open, spatial_ref_closed, verbose,
    )

    n_ticks = len(y)
    n_raw_inputs = X_raw.shape[1]

    # Add derived features
    X = _add_derived_features(X_raw, has_spatial)
    feature_names = _get_feature_names(has_spatial)

    if verbose:
        print(f"\n[CATBOOST] Extended features: {X.shape[1]} ({', '.join(feature_names)})")
        for i, name in enumerate(feature_names):
            if i >= n_raw_inputs:  # Only print derived features
                print(f"    {name}: mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}, "
                      f"range=[{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    # Train CatBoost
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function='Logloss',
        verbose=False,
        random_seed=42,
        auto_class_weights='Balanced',
    )

    if verbose:
        print(f"[CATBOOST] Training with {iterations} iterations, depth={depth}, lr={learning_rate}...")

    from catboost import Pool
    train_pool = Pool(X, y.astype(int), feature_names=feature_names)
    model.fit(train_pool)

    # Evaluate
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(float)
    accuracy = float(np.mean(preds == y))

    open_mask = y == 0
    closed_mask = y == 1
    acc_open = float(np.mean(preds[open_mask] == 0)) if open_mask.any() else 0.0
    acc_closed = float(np.mean(preds[closed_mask] == 1)) if closed_mask.any() else 0.0

    n_trees = model.tree_count_

    if verbose:
        print(f"\n[CATBOOST] Training complete:")
        print(f"  Overall accuracy: {accuracy:.1%}")
        print(f"  OPEN accuracy:    {acc_open:.1%}")
        print(f"  CLOSED accuracy:  {acc_closed:.1%}")
        print(f"  Trees:            {n_trees}")

        # Feature importance
        importances = model.get_feature_importance()
        print(f"  Feature importance:")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            print(f"    {feature_names[idx]:<15s} {importances[idx]:6.1f}%")

    # Save model to bytes
    import tempfile
    import os

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.cbm')
    os.close(tmp_fd)
    try:
        model.save_model(tmp_path)
        with open(tmp_path, 'rb') as f:
            model_bytes = f.read()
    finally:
        os.unlink(tmp_path)

    model_data = {
        'model_bytes': model_bytes,
        'n_inputs': n_raw_inputs,
        'has_spatial': has_spatial,
        'accuracy': accuracy,
        'accuracy_open': acc_open,
        'accuracy_closed': acc_closed,
        'n_training_samples': n_ticks,
        'n_trees': n_trees,
        'feature_names': feature_names,
    }

    return model_data
