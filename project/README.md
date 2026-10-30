# Physics-Informed Neural Networks for Kinetics-Constrained Biochar Production
**Closed-Loop Optimization Across Diverse Biomass Feedstocks**

---

## 1. Architectural Design & Separation of Concerns

This codebase implements two completely independent, decoupled tracks with **zero parameter sharing, separate models, independent data pipelines, and isolated configs**:

```
project/
├── utils/                     # Stateless shared physics & math library (no trainable parameters)
│   ├── transforms.py          # Compositional simplex log-ratios (ILR & CLR)
│   ├── thermal_profile.py     # Explicit piecewise ramp-and-hold T(t) trajectory
│   ├── ode_integrator.py      # Differentiable Runge-Kutta 4th-order multi-component ODE solver
│   ├── energy_physics.py      # First-principles thermodynamic net enthalpy balance
│   └── metrics.py             # Point metrics (RMSE, MAE, R²) and UQ metrics (PICP, MPIW, NLL)
│
├── real_life/                 # Production-oriented track (cheap, rapidly measurable inputs)
│   ├── data/                  # Block 1 (Biopolymers) or Block 2 (Proximate fallback) + Process
│   ├── model/                 # Feedstock/Process branch fusion, heteroscedastic head, ODE loss
│   ├── acquisition/           # Cost-aware batch Upper Confidence Bound (UCB)
│   ├── checkpoints/           # Saved Deep Ensemble model weights
│   ├── train.py               # Deep Ensemble training loop (M >= 5)
│   ├── evaluate.py            # Held-out generalization, energy calculation & active recommendations
│   └── config.yaml            # Configurable runtime parameters
│
├── research/                  # Scientific exploration track (high-dimensional multi-block inputs)
│   ├── data/                  # Blocks 1, 2, 3, 4 + Moisture (multi-target masking)
│   ├── model/                 # Multi-block fusion, calibrated kinetic ODE, multi-property heads
│   ├── acquisition/           # Multi-objective Pareto frontier / Expected Hypervolume Improvement
│   ├── checkpoints/           # Saved research ensemble checkpoints
│   ├── train.py               # End-to-end multi-task training
│   ├── evaluate.py            # Multi-target held-out benchmarking & Pareto recommendations
│   └── config.yaml            # Configurable runtime parameters
│
├── evaluation/                # Cross-track benchmarking against standard ML baselines
│   ├── baselines.py           # Random Forest, XGBoost, GPR, MC Dropout MLP
│   └── run_benchmarks.py      # Side-by-side comparison on identical held-out feedstock splits
│
└── tests/
    └── test_all_components.py # Comprehensive unit test suite
```

---

## 2. Mathematical Formulations

### 2.1 Compositional Simplexes (No Raw Collinear Percentages)
Biomass composition is strictly simplex-constrained: $\sum x_i = 1.0$.
We transform compositions via **Centered Log-Ratio (CLR)** or **Isometric Log-Ratio (ILR)**:
$$\text{clr}(x)_i = \ln \left( \frac{x_i}{g(x)} \right), \quad g(x) = \left( \prod_{j=1}^D x_j \right)^{1/D}$$
Inverse transforms map predicted or surrogate simplexes back to valid percentages without sum-to-100 violations.

### 2.2 Explicit Thermal Trajectory $T(t)$
Pyrolysis conditions $(T_{\text{peak}}, \text{RT}, \text{HR})$ define a continuous thermal profile:
$$T(t) = \begin{cases} T_0 + \text{HR} \cdot t & \text{for } 0 \le t \le t_{\text{ramp}} \\ T_{\text{peak}} & \text{for } t_{\text{ramp}} < t \le t_{\text{ramp}} + \text{RT} \end{cases}$$

### 2.3 ODE Kinetic Devolatilization Layer
Distributed multi-component parallel reaction model for Hemicellulose, Cellulose, Lignin, and Extractives:
$$\frac{d\alpha_i(t)}{dt} = A_i \exp\left( -\frac{E_{a,i}}{R \cdot T(t)} \right) (1 - \alpha_i(t))^{n_i}$$
Integrated differentiably across the thermal profile via Runge-Kutta 4th order (RK4). The kinetic char yield is:
$$Y_{\text{kinetic}} = \sum_{i=1}^4 w_i \left[ (1 - \alpha_i) + \alpha_i \cdot Y_{\text{char}, \infty, i} \right] (1 - w_{\text{ash}}) + w_{\text{ash}}$$

### 2.4 First-Principles Energy Balance
Net specific energy requirement per kg dry biomass input ($E_{\text{net}}$):
$$E_{\text{net}} = E_{\text{sensible}} + E_{\text{evap}} + \Delta H_{\text{reaction}} - \eta_{\text{rec}} E_{\text{syngas}}$$

### 2.5 Physics Loss & Monotonicity Constraint
$$\mathcal{L} = e^{-s_{\text{data}}} \mathcal{L}_{\text{NLL}} + \frac{1}{2} s_{\text{data}} + e^{-s_{\text{kin}}} \mathcal{L}_{\text{kinetics}} + \frac{1}{2} s_{\text{kin}} + \lambda_{\text{mono}} \frac{1}{K} \sum_{k=1}^K \text{ReLU}\left( \frac{\partial \hat{Y}}{\partial T_k} \right)^2$$
Enforces that biochar yield decreases monotonically as temperature increases ($\frac{\partial Y}{\partial T} \le 0$) over synthetic collocation points.

---

## 3. How to Run

### Step 1: Run the Unit Tests
```bash
python project/tests/test_all_components.py
```

### Step 2: Real-Life Track Training & Evaluation
```bash
# 1. Train the Deep Ensemble (5 members with independent random seeds)
python -u project/real_life/train.py

# 2. Evaluate generalization and generate active learning recommendations
python project/real_life/evaluate.py
```

#### Toggling to Proximate Fallback (No Code Rewrite):
In `project/real_life/config.yaml`, change:
```yaml
data:
  feature_mode: "proximate" # Automatically switches from Block 1 to Proximate Block 2
```

### Step 3: Research Track Multi-Target Modeling & Pareto Optimization
```bash
# 1. Train multi-target Research Ensemble with learnable kinetic priors
python -u project/research/train.py

# 2. Evaluate multi-target predictions and multi-objective Pareto frontier
python project/research/evaluate.py
```

### Step 4: Run Unified Cross-Model Benchmarking
```bash
python project/evaluation/run_benchmarks.py
```
This trains the PINN Deep Ensemble alongside Random Forest, XGBoost, Gaussian Process, and MC Dropout MLP on the exact same held-out feedstock splits, reporting $R^2$, RMSE, MAE, 95% Coverage (PICP), and Prediction Interval Width (MPIW).
