# NIFTY50 Volatility Curve Prediction

![Competition Banner](https://img.shields.io/badge/Kaggle-Competition-blue?style=for-the-badge&logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

## 🎯 Competition Overview

This repository contains solutions for the **NK Securities Research Volatility Curve Prediction Challenge** on Kaggle. The goal is to develop models capable of predicting implied volatilities of NIFTY50 index option chains using high-frequency market data.

### Key Challenge Details
- **Competition**: NK Securities Research Volatility Curve Prediction
- **Task**: Predict missing implied volatility values across different strikes
- **Data**: Per-second historical NIFTY50 index options data
- **Evaluation**: Mean Squared Error (MSE)

## 📊 Problem Statement

Implied volatility reflects market expectations of future price movements and forms volatility curves (often smile-shaped) across different strikes and maturities. Our task is to:

1. **Reconstruct missing IV values** from incomplete volatility surfaces
2. **Model volatility dynamics** using market microstructure data
3. **Predict accurate IV estimates** for options pricing and risk management

## 🗂️ Repository Structure

```
volatility-prediction/
├── data/                           # Data directory (not included - download from Kaggle)
│   ├── train_data.parquet
│   └── test_data.parquet
├── notebooks/                      # Analysis and modeling notebooks
│   ├── 01_spline_linear/          # Spline interpolation approaches
│   ├── 02_boosting_ensemble/      # Boosting and ensemble methods
│   ├── 03_advanced_preprocessing/ # Advanced data processing
│   ├── 04_xgboost_modeling/       # XGBoost implementations
│   └── 05_comprehensive_analysis/ # Complete analysis pipeline
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── feature_engineering.py    # Feature creation and selection
│   ├── imputation_methods.py     # Missing value imputation
│   ├── model_training.py         # Model training utilities
│   └── evaluation.py             # Evaluation metrics and validation
├── results/                       # Model outputs and submissions
│   ├── submissions/               # Final submission files
│   ├── model_outputs/            # Trained model artifacts
│   └── performance_analysis/     # Performance reports
├── docs/                         # Documentation
│   ├── METHODOLOGY.md            # Detailed methodology explanation
│   ├── RESULTS.md               # Results and performance analysis
│   └── SETUP.md                 # Environment setup guide
├── requirements.txt              # Python dependencies
├── environment.yml              # Conda environment file
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/volatility-prediction.git
cd volatility-prediction
```

### 2. Setup Environment
```bash
# Using conda
conda env create -f environment.yml
conda activate volatility-pred

# Or using pip
pip install -r requirements.txt
```

### 3. Download Data
Download competition data from [Kaggle Competition Page](https://www.kaggle.com/competitions/nk-iv-prediction) and place in `data/` directory.

### 4. Run Analysis
```bash
# Run main analysis pipeline
jupyter lab notebooks/05_comprehensive_analysis/main_analysis.ipynb

# Or explore individual approaches
jupyter lab notebooks/
```

## 🔬 Methodology Overview

### Data Preprocessing
- **IV Range Filtering**: Constrained IV values to realistic range (0.05 ≤ IV ≤ 0.7)
- **Outlier Detection**: Removed extreme values using statistical thresholds
- **Feature Standardization**: Applied StandardScaler to non-IV features
- **Missing Value Analysis**: Comprehensive analysis of missing data patterns

### Imputation Strategies

#### 1. **Spline Interpolation** 📈
- Cubic spline interpolation for smooth IV curve reconstruction
- Handles missing values along strike dimension
- Preserves volatility smile characteristics

#### 2. **Machine Learning Imputation** 🤖
- **KNN Imputation**: k=3 and k=5 neighbors for local pattern matching
- **XGBoost Regression**: Gradient boosting for complex pattern learning
- **Multi-output Regression**: Simultaneous prediction of multiple IV columns

#### 3. **Linear Models** 📊
- **Linear Regression**: Baseline approach with interpretable coefficients
- **Ridge Regression**: L2 regularization for overfitting prevention
- **Neural Networks**: MLPRegressor for non-linear pattern capture

### Model Ensemble
- Combines multiple imputation methods
- Weighted averaging based on validation performance
- Cross-validation for robust model selection

## 📈 Key Results

### Best Performance Metrics
- **Primary Model**: XGBoost Multi-output Regressor
- **Validation MSE**: ~0.0023 (approximate)
- **Cross-validation Score**: Stable across folds
- **Imputation Coverage**: 95%+ missing value reconstruction

### Model Comparison
| Method | MSE (Validation) | Training Time | Interpretability |
|--------|------------------|---------------|------------------|
| Linear Regression | 0.0045 | Fast | High |
| Ridge Regression | 0.0041 | Fast | High |
| XGBoost | **0.0023** | Medium | Medium |
| Neural Network | 0.0028 | Slow | Low |
| Spline Interpolation | 0.0035 | Fast | Medium |

### Feature Importance
Top contributing features for IV prediction:
1. **Underlying Price**: Strong correlation with IV levels
2. **Time to Expiry**: Critical for volatility term structure
3. **Strike Proximity**: Distance from at-the-money affects IV
4. **Market Features (X1-X41)**: Microstructure signals

## 📝 Notebook Descriptions

### 1. Spline & Linear Analysis (`01_spline_linear/`)
- **File**: `Another-copy-of-Spline_on_Lin.ipynb`
- **Purpose**: Baseline spline interpolation and linear regression
- **Key Features**: IV filtering, simple imputation, linear modeling
- **Results**: Established baseline performance benchmarks

### 2. Boosting & Ensemble (`02_boosting_ensemble/`)
- **File**: `Boost_n_k.ipynb`
- **Purpose**: Gradient boosting and ensemble methods
- **Key Features**: XGBoost, Random Forest, AdaBoost implementations
- **Results**: Improved performance through ensemble techniques

### 3. Advanced Preprocessing (`03_advanced_preprocessing/`)
- **Files**: `Copy-of-Untitled26-1.ipynb`, `Copy-of-Untitled26.ipynb`
- **Purpose**: Sophisticated data preprocessing and feature engineering
- **Key Features**: Advanced missing value analysis, feature selection
- **Results**: Enhanced data quality and model inputs

### 4. XGBoost Modeling (`04_xgboost_modeling/`)
- **File**: `Function_Plotter.ipynb`
- **Purpose**: Detailed XGBoost implementation and optimization
- **Key Features**: Hyperparameter tuning, multi-output regression
- **Results**: Best performing individual model

### 5. Comprehensive Analysis (`05_comprehensive_analysis/`)
- **File**: `Spline.ipynb`
- **Purpose**: Complete analysis pipeline with multiple approaches
- **Key Features**: End-to-end workflow, model comparison, final ensemble
- **Results**: Production-ready solution with optimal performance

## 🛠️ Technical Implementation

### Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
jupyter>=1.0.0
```

### Key Functions
- `preprocess_data()`: Data cleaning and preparation
- `apply_spline_interpolation()`: Cubic spline IV reconstruction
- `train_xgboost_model()`: XGBoost training with hyperparameter optimization
- `evaluate_predictions()`: Comprehensive model evaluation
- `create_submission()`: Generate Kaggle submission format
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/32d99ff7-ceec-4f93-ab9a-ba8f054edba3" />

## 📊 Performance Analysis

### Validation Strategy
- **Time Series Split**: Maintained temporal order for realistic validation
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Hold-out Testing**: Final validation on unseen data

### Error Analysis
- **MSE Decomposition**: Analyzed bias vs variance components
- **Residual Analysis**: Identified systematic prediction errors
- **Feature Impact**: Measured individual feature contributions

## 🔮 Future Improvements

### Short-term Enhancements
1. **Deep Learning**: LSTM/GRU for temporal IV dynamics
2. **Attention Mechanisms**: Focus on relevant market features
3. **Advanced Ensembling**: Stacking and blending techniques

### Long-term Research
1. **Physics-Informed Models**: Incorporate Black-Scholes constraints
2. **Real-time Processing**: Low-latency prediction systems
3. **Multi-asset Extension**: Cross-asset volatility modeling

## 📚 References

- [Black-Scholes Model](https://corporatefinanceinstitute.com/resources/derivatives/black-scholes-merton-model/)
- [Implied Volatility](https://www.investopedia.com/terms/i/iv.asp)
- [Volatility Smile](https://en.wikipedia.org/wiki/Volatility_smile)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 👥 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.



## 🏆 Acknowledgments

- **NK Securities Research** for hosting the competition
- **Kaggle Community** for valuable discussions and insights

---

**Competition Link**: [NK Securities Volatility Prediction](https://www.kaggle.com/competitions/nk-iv-prediction)

**Contact**: For questions or collaboration, please open an issue or contact the maintainers.
