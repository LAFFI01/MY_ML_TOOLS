# 🔮 OPTUNA COMPLETE PARAMETER REFERENCE

## Overview

Optuna is a **Bayesian Optimization** framework that intelligently searches hyperparameter space. Instead of exhaustively trying all combinations (Grid Search) or random sampling (Random Search), Optuna learns which hyperparameters work best and focuses its search there. Result: **Better parameters in fewer trials**.

---

## ⚙️ ALL Optuna Parameters

### 1. **`search_type='optuna'`** ← REQUIRED
Tells the toolkit to use Optuna instead of GridSearch/RandomSearch.

```python
search_type='optuna'  # Must use this to enable Optuna
```

---

### 2. **`optuna_n_trials=50`** ← Number of Trials
How many hyperparameter combinations to try.

| Value | Use Case | Time |
|-------|----------|------|
| 10 | Quick demo | < 1 min |
| 50 | Standard tuning | 5-10 min |
| 100 | Thorough tuning | 10-30 min |
| 200+ | Exhaustive search | 30+ min |

```python
# Try 50 different hyperparameter combinations
optuna_n_trials=50
```

---

### 3. **`optuna_timeout=60`** ← Time Limit (Seconds)
Maximum seconds to spend tuning. Stops early if limit reached (even if n_trials not complete).

| Value | Use Case |
|-------|----------|
| `None` | Run all n_trials (default) |
| 30 | Quick 30-second search |
| 300 | 5-minute search |
| 3600 | 1-hour search |

```python
# Run up to 100 trials OR 5 minutes, whichever comes first
optuna_n_trials=100
optuna_timeout=300  # 5 minutes
```

**🎯 When to use:** When you're on a laptop/Kaggle and need results quickly.

---

### 4. **`optuna_storage='sqlite:///optuna.db'`** ← Persistent Storage
Save optimization history to database so you can resume later.

| Option | Behavior | Use Case |
|--------|----------|----------|
| `None` | In-memory only | Quick experiments |
| `'sqlite:///study.db'` | Local SQLite file | Resume interrupted runs |
| `'redis://localhost'` | Redis server | Distributed across machines |

```python
# Save to local file (can resume later)
optuna_storage='sqlite:///./my_optuna_study.db'

# Save to absolute path (Kaggle notebooks)
optuna_storage='sqlite:////tmp/my_study.db'

# Distributed Redis (multiple machines)
optuna_storage='redis://your-redis-server:6379'
```

**🎯 Benefits:**
- Interrupt and resume without losing progress
- Run same study from multiple notebooks
- Compare trials across different runs

---

### 5. **`optuna_study_name='my_study'`** ← Study Name
Name for the optimization study. Used to resume and organize experiments.

```python
# First run
results = evaluate_and_plot_models(
    ...,
    optuna_study_name='housing_regression_v1',
    optuna_storage='sqlite:///studies.db'
)

# Later: Resume same study
results = evaluate_and_plot_models(
    ...,
    optuna_study_name='housing_regression_v1',  # ← Same name!
    optuna_storage='sqlite:///studies.db'       # ← Same location
)
# Will add 50 MORE trials to the previous 50 (100 total)
```

**🎯 When to use:** When you want to continue tuning instead of starting fresh.

---

### 6. **`distributed_tuning=True`** ← Parallel Notebooks
Enable parallel tuning across multiple notebooks/machines sharing the SAME database.

```python
# ===== NOTEBOOK 1 =====
results = evaluate_and_plot_models(
    optuna_n_trials=30,
    optuna_storage='sqlite:///kaggle_shared.db',  # SAME database
    optuna_study_name='kaggle_competition',        # SAME study name
    distributed_tuning=True,                       # Enable parallel
)

# ===== NOTEBOOK 2 =====  
results = evaluate_and_plot_models(
    optuna_n_trials=30,  # Another 30 trials simultaneously
    optuna_storage='sqlite:///kaggle_shared.db',
    optuna_study_name='kaggle_competition',
    distributed_tuning=True,
)

# ===== NOTEBOOK 3 =====
results = evaluate_and_plot_models(
    optuna_n_trials=30,  # Another 30 trials simultaneously
    optuna_storage='sqlite:///kaggle_shared.db',
    optuna_study_name='kaggle_competition',
    distributed_tuning=True,
)

# Result: 90 trials running in parallel across 3 notebooks = 3x speedup! 🚀
```

**🎯 Perfect for:** Kaggle competitions where you have multiple notebooks running in parallel.

---

## 📋 Complete Parameter Combinations

### **Scenario 1: Quick Local Experiment**
```python
results = evaluate_and_plot_models(
    models=models,
    X=X, y=y,
    search_type='optuna',           # Use Optuna
    optuna_n_trials=20,             # Quick 20 trials
    optuna_timeout=None,            # No time limit
    optuna_storage=None,            # In-memory
    optuna_study_name=None,         # Auto-generated
    distributed_tuning=False,       # Single machine
    ...
)
```

---

### **Scenario 2: Resume Interrupted Run**
```python
# First run
results = evaluate_and_plot_models(
    ...,
    optuna_n_trials=50,
    optuna_storage='sqlite:///my_study.db',
    optuna_study_name='experiment_1',
    distributed_tuning=False,
)
# Interrupted after 20 trials!

# Later: Resume and continue
results = evaluate_and_plot_models(
    ...,
    optuna_n_trials=50,  # Try 50 MORE
    optuna_storage='sqlite:///my_study.db',     # ← SAME
    optuna_study_name='experiment_1',           # ← SAME
    distributed_tuning=False,
)
# Now has 70 total trials (20 + 50)
```

---

### **Scenario 3: Distributed Kaggle Tuning (3x Speedup)**
```python
# Run in 3 parallel Kaggle notebooks

results = evaluate_and_plot_models(
    models=models,
    X=X, y=y,
    search_type='optuna',
    optuna_n_trials=50,                                    # 50 per notebook
    optuna_timeout=600,                                    # 10 min per notebook
    optuna_storage='sqlite:///./kaggle_shared_db.db',     # 💾 SHARED
    optuna_study_name='kaggle_competition_2026',          # 📝 SHARED NAME
    distributed_tuning=True,                               # ✅ PARALLEL MODE
    ...
)

# Each notebook runs 50 trials simultaneously
# If all 3 notebooks: 150 trials in parallel = equivalent to ~50 serial trials! ✨
```

---

### **Scenario 4: Time-Based Search (Kaggle Timeout)**
```python
results = evaluate_and_plot_models(
    ...,
    search_type='optuna',
    optuna_n_trials=10000,        # "Try as many as possible"
    optuna_timeout=120,           # BUT STOP after 2 minutes ⏱️
    optuna_storage=None,
    optuna_study_name=None,
    distributed_tuning=False,
    ...
)
# Will run trials until 2 minutes elapsed, then stop
```

---

## 🎯 Decision Tree: Which Parameters to Use?

```
❓ What's your use case?

├─ Quick test (< 2 min)?
│  └─ search_type='optuna'
│     optuna_n_trials=10
│     optuna_timeout=None
│     All others: None/False
│
├─ Heavy tuning on local machine?
│  └─ search_type='optuna'
│     optuna_n_trials=100
│     optuna_timeout=None
│     optuna_storage='sqlite:///study.db'
│     optuna_study_name='v1'
│     distributed_tuning=False
│
├─ Kaggle: Want to resume interrupted run?
│  └─ search_type='optuna'
│     optuna_n_trials=100
│     optuna_timeout=300  # 5 min safety
│     optuna_storage='sqlite:///kaggle_study.db'
│     optuna_study_name='kaggle_run_1'
│     distributed_tuning=False
│
└─ Kaggle: 3 parallel notebooks (3x speed)?
   └─ search_type='optuna'
      optuna_n_trials=50          # Per notebook
      optuna_timeout=600          # 10 min per notebook
      optuna_storage='sqlite:///kaggle_shared.db'   # SHARED ✨
      optuna_study_name='competition'               # SHARED ✨
      distributed_tuning=True                        # PARALLEL ✨
```

---

## 📊 Optuna vs GridSearch vs RandomSearch

| Feature | Optuna | GridSearch | RandomSearch |
|---------|--------|-----------|--------------|
| **Trials Needed** | 50-100 | 100-500+ | 100-200 |
| **Speed** | ⚡⚡⚡ Fast | ⚠️ Slow | ⚡ Medium |
| **Quality** | 🏆 Best | ✅ Good | ⚠️ Random |
| **Persistence** | ✅ Yes | ❌ No | ❌ No |
| **Distributed** | ✅ Yes | ❌ No | ❌ No |
| **How it works** | Bayesian learning | Exhaustive | Random sampling |

---

## 💡 Pro Tips

### ✅ DO:
```python
# ✅ Good: Persistent + shared
optuna_storage='sqlite:///study.db'
optuna_study_name='my_experiment'
distributed_tuning=True

# ✅ Good: Balance trials and timeout
optuna_n_trials=100
optuna_timeout=600  # Safety net

# ✅ Good: Meaningful study names
optuna_study_name='housing_xgboost_v3'
```

### ❌ DON'T:
```python
# ❌ Bad: Very high trials without timeout (infinite wait)
optuna_n_trials=10000
optuna_timeout=None

# ❌ Bad: Different study names (separate studies)
optuna_study_name='study1'  # Run 1
optuna_study_name='study2'  # Run 2 (can't resume!)

# ❌ Bad: Wrong storage path syntax
optuna_storage='sqlite://optuna.db'  # Missing slash!
# Correct: optuna_storage='sqlite:///optuna.db'
```

---

## 🚀 Usage Files

Created complete examples:
- **simple**: `example_optuna_advanced.py` in `/examples/`

Run examples:
```bash
cd /home/laffi/CODE\ /MY_tools
source .venv/bin/activate

# Run all examples
python examples/example_optuna_advanced.py

# Or run individually
python -c "from examples.example_optuna_advanced import example_1_regression_basic_optuna; example_1_regression_basic_optuna()"
```

---

## 📈 Example Output

```
[Ridge] Running 🔮 Optuna Bayesian Optimization (n_trials=50)
[Ridge] Best Optuna Trial #23: Score=0.9234
   └─ Best params: {'model__alpha': 0.47}

[XGBoost] Running 🔮 Optuna Bayesian Optimization (n_trials=50)  
[XGBoost] Best Optuna Trial #41: Score=0.9512
   └─ Best params: {'model__learning_rate': 0.087, 'model__max_depth': 7}
```

Optuna found:
- Ridge: `alpha=0.47` is optimal
- XGBoost: `learning_rate=0.087, max_depth=7` is optimal

These become the final model parameters!

---

## ❓ FAQ

**Q: Can I use Optuna with GridSearch?**
A: No, use either `search_type='optuna'` or `search_type='grid'`, not both.

**Q: Will resuming add MORE trials or restart?**
A: It continues from where it left off (additive).

**Q: Can distributed_tuning work without storage?**
A: No, you MUST use `optuna_storage` with `distributed_tuning=True`.

**Q: What if multiple notebooks have different `optuna_study_name`?**
A: They create separate studies (don't share trials).

**Q: Is Redis necessary for Kaggle?**
A: No! SQLite is perfect for Kaggle. Redis is for cloud clusters.

