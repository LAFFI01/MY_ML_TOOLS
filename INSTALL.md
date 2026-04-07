# Installation Guide

Quick and comprehensive guide to install and start using **ML Toolkit**.

---

## 📦 Installation Methods

### **Method 1: From GitHub (Recommended)**

```bash
# Basic installation
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git

# With development tools
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git[dev]

# With documentation tools
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git[docs]

# With everything
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git[dev,docs]
```

### **Method 2: Clone & Install Locally**

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MY_ML.git
cd MY_ML

# Install in development mode (editable)
pip install -e .

# Or with extras
pip install -e .[dev]
```

### **Method 3: From PyPI (After Publishing)**

```bash
pip install my-ml-toolkit
```

---

## ✅ Verify Installation

```bash
# Test import
python -c "from my_ml_toolkit import evaluate_and_plot_models; print('✅ Installation successful!')"

# Or check version
python -c "import my_ml_toolkit; print(f'Version: {my_ml_toolkit.__version__}')"
```

---

## 📋 Automatic Dependencies

When you install the package, these libraries are **automatically installed**:

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation |
| numpy | ≥1.24.0 | Numerical computing |
| matplotlib | ≥3.7.0 | Visualization & plots |
| scikit-learn | ≥1.5.0 | Machine learning models |
| imbalanced-learn | ≥0.11.0 | Handle imbalanced data |
| joblib | ≥1.0.0 | Parallel processing |

### Optional Dependencies

**Development Tools:**
```bash
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git[dev]
```
Includes: pytest, black, flake8, mypy, isort, pre-commit, build, twine

**Documentation:** 
```bash
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git[docs]
```
Includes: sphinx, sphinx-rtd-theme

---

## 🛠️ System Requirements

- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **pip**: 20.0 or higher
- **OS**: Linux, macOS, or Windows

### Check Python Version

```bash
python --version  # Should be 3.8+
pip --version     # Should be 20.0+
```

---

## 🐛 Troubleshooting

### Issue: Command not found - pip

**Solution:**
```bash
# Try python3 instead
python3 -m pip install git+https://github.com/YOUR_USERNAME/MY_ML.git
```

### Issue: Permission denied

**Solution:**
```bash
# For user installation
pip install --user git+https://github.com/YOUR_USERNAME/MY_ML.git

# Or use virtual environment (recommended)
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install git+https://github.com/YOUR_USERNAME/MY_ML.git
```

### Issue: Module not found

**Solution:**
```bash
# Verify installation
pip list | grep my-ml-toolkit

# Try reinstalling
pip install --force-reinstall git+https://github.com/YOUR_USERNAME/MY_ML.git
```

### Issue: Git not installed

**Solution:**
```bash
# Install git: https://git-scm.com/downloads
# Or use requirements.txt method instead
```

---

## 📝 requirements.txt Method

Create a `requirements.txt` file:

```txt
git+https://github.com/YOUR_USERNAME/MY_ML.git@main
```

Then install:

```bash
pip install -r requirements.txt
```

---

## 🐳 Using with Docker

```dockerfile
FROM python:3.11-slim

RUN pip install git+https://github.com/YOUR_USERNAME/MY_ML.git

CMD ["python"]
```

Build and run:

```bash
docker build -t my-ml-toolkit .
docker run -it my-ml-toolkit python
```

---

## ✨ Next Steps

After installation:

1. **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
2. **Full Examples**: Check `examples/` folder
3. **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)
4. **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📚 Additional Resources

- [GitHub Repository](https://github.com/YOUR_USERNAME/MY_ML)
- [Issue Tracker](https://github.com/YOUR_USERNAME/MY_ML/issues)
- [Development Guide](DEVELOPMENT.md)

---

Need help? Open an issue on GitHub! 🤝
