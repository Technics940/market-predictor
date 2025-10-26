# 🔧 Setup Guide - Market Predictor

Complete installation and setup instructions for the Market Predictor program.

---

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.8 or higher** installed on your system
2. **pip** (Python package manager)
3. **Internet connection** (to download packages and market data)

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/)

---

## 🚀 Installation Methods

### Method 1: Using requirements.txt (Recommended)

This is the easiest and fastest method.

1. **Navigate to the project directory**:
```bash
cd /path/to/market_predictor
```

2. **Install all dependencies at once**:
```bash
pip install -r requirements.txt
```

That's it! You're ready to go.

---

### Method 2: Manual Installation

Install each package individually:

```bash
# Core data fetching library
pip install yfinance

# Data manipulation
pip install pandas numpy

# Machine learning
pip install scikit-learn

# Visualization
pip install matplotlib

# Optional but recommended
pip install seaborn
```

---

### Method 3: Using a Virtual Environment (Best Practice)

Virtual environments keep your project dependencies isolated from other Python projects.

#### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

To deactivate the virtual environment later:
```bash
deactivate
```

---

## ✅ Verify Installation

Run this command to check if all packages are installed:

```bash
python -c "import yfinance, pandas, numpy, sklearn, matplotlib; print('✓ All packages installed successfully!')"
```

If you see the success message, you're ready to use the program!

---

## 🎯 Quick Test

Test the installation by running a simple prediction:

```bash
python simple_example.py
```

This will:
- Fetch Apple (AAPL) stock data
- Train a prediction model
- Generate predictions for the next 7 days
- Save a chart as `AAPL_predictions.png`

Expected output should include:
- ✓ Successfully fetched data
- ✓ Model training complete
- ✓ Future predictions
- ✓ Plot saved

---

## 🐛 Troubleshooting

### Problem 1: "pip: command not found"

**Solution:**
```bash
# Try using pip3 instead
pip3 install -r requirements.txt

# Or use python -m pip
python -m pip install -r requirements.txt
python3 -m pip install -r requirements.txt
```

---

### Problem 2: Permission denied errors

**Solution (Linux/macOS):**
```bash
# Install for current user only
pip install --user -r requirements.txt
```

**Solution (Windows - Run as Administrator):**
- Right-click Command Prompt
- Select "Run as Administrator"
- Run the pip install command

---

### Problem 3: "No module named 'sklearn'"

The package is called `scikit-learn` but imported as `sklearn`.

**Solution:**
```bash
pip install scikit-learn
```

---

### Problem 4: SSL Certificate errors

**Solution:**
```bash
# Try upgrading pip first
pip install --upgrade pip

# Then retry installation
pip install -r requirements.txt

# If still failing, try with --trusted-host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

### Problem 5: "Microsoft Visual C++ is required" (Windows)

Some packages require compilation. 

**Solution:**
Download and install "Microsoft C++ Build Tools" from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Or install pre-built wheels:
```bash
pip install --only-binary :all: scikit-learn
```

---

### Problem 6: Slow installation

**Solution:**
Some packages like scikit-learn are large. This is normal. Be patient.

You can see progress with:
```bash
pip install -v -r requirements.txt
```

---

### Problem 7: "externally-managed-environment" error (Linux)

Modern Linux distributions use system Python protection.

**Solution 1 - Use virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Solution 2 - Install with --break-system-packages:**
```bash
pip install --break-system-packages -r requirements.txt
```

---

## 🔄 Updating Packages

To update all packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```

To update a specific package:
```bash
pip install --upgrade yfinance
```

---

## 🧪 Testing Your Setup

### Test 1: Import All Libraries
```python
python3
>>> import yfinance as yf
>>> import pandas as pd
>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> import matplotlib.pyplot as plt
>>> print("✓ All imports successful!")
>>> exit()
```

### Test 2: Fetch Sample Data
```python
python3
>>> import yfinance as yf
>>> data = yf.Ticker("AAPL").history(period="5d")
>>> print(f"✓ Fetched {len(data)} days of data")
>>> exit()
```

### Test 3: Run Quick Prediction
```bash
python simple_example.py
```

---

## 📦 Package Versions

The program is tested with these versions:

| Package | Minimum Version | Tested Version |
|---------|----------------|----------------|
| Python | 3.8 | 3.11 |
| yfinance | 0.2.28 | 0.2.32 |
| pandas | 1.5.0 | 2.1.0 |
| numpy | 1.23.0 | 1.26.0 |
| scikit-learn | 1.2.0 | 1.3.0 |
| matplotlib | 3.6.0 | 3.8.0 |

To check installed versions:
```bash
pip list | grep -E "yfinance|pandas|numpy|scikit-learn|matplotlib"
```

Or:
```bash
pip show yfinance pandas numpy scikit-learn matplotlib
```

---

## 🌐 Network Requirements

The program requires internet access to:

1. **Download market data** from Yahoo Finance (yfinance API)
2. **Install Python packages** from PyPI

If you're behind a proxy, configure pip:
```bash
pip install --proxy http://user:password@proxy-server:port -r requirements.txt
```

---

## 💾 Disk Space Requirements

- Python packages: ~500 MB
- Program files: ~50 KB
- Generated data/charts: ~1 MB per prediction

**Total: ~500-600 MB**

---

## 🖥️ System Requirements

### Minimum:
- **OS**: Windows 7+, macOS 10.13+, Linux (any modern distro)
- **RAM**: 2 GB
- **CPU**: Any modern processor
- **Disk**: 1 GB free space

### Recommended:
- **OS**: Windows 10+, macOS 11+, Ubuntu 20.04+
- **RAM**: 4 GB or more
- **CPU**: Multi-core processor
- **Disk**: 2 GB free space

---

## 🐍 Python Environment Options

### Option 1: System Python
Install packages globally (simplest but not recommended for multiple projects)

### Option 2: Virtual Environment (venv)
Built-in, lightweight, perfect for this project
```bash
python -m venv venv
```

### Option 3: Conda
If you use Anaconda/Miniconda:
```bash
conda create -n market_predictor python=3.11
conda activate market_predictor
pip install -r requirements.txt
```

### Option 4: Poetry
For advanced users:
```bash
poetry install
poetry run python market_predictor.py
```

---

## 🎓 Next Steps After Setup

1. ✅ **Verify installation** - Run test commands above
2. 📖 **Read the documentation** - Check `README.md` and `QUICKSTART.md`
3. 🚀 **Run your first prediction** - Try `simple_example.py`
4. 🔧 **Customize** - Edit scripts to predict your favorite stocks
5. 📊 **Analyze results** - Review generated charts and metrics

---

## 📞 Getting Help

### Installation Issues
1. Check the troubleshooting section above
2. Verify your Python version is 3.8+
3. Try using a virtual environment
4. Check your internet connection

### Package Errors
1. Update pip: `pip install --upgrade pip`
2. Clear pip cache: `pip cache purge`
3. Retry installation

### Runtime Errors
1. Ensure all packages are installed
2. Check if ticker symbol is valid
3. Verify internet connection
4. Try a different time period or ticker

---

## 🔐 Security Notes

- No API keys required (yfinance is free and open)
- No personal data collected
- All data processing is local
- Internet only used for downloading data

---

## 🆘 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'yfinance'` | Package not installed | `pip install yfinance` |
| `ImportError: cannot import name 'RandomForestRegressor'` | scikit-learn not installed | `pip install scikit-learn` |
| `No data found for ticker` | Invalid ticker or no internet | Check ticker spelling |
| `HTTPError: 404 Client Error` | Yahoo Finance API issue | Try again later |

---

## ✨ You're All Set!

Once installation is complete, you can:

```bash
# Run the main program
python market_predictor.py

# Run simple example
python simple_example.py

# Compare multiple stocks
python batch_predictor.py

# Use utility functions
python market_utils.py
```

Happy predicting! 📈🚀

---

**Need more help?**
- Read: `README.md` for full documentation
- Read: `QUICKSTART.md` for quick examples
- Check: Source code comments for details