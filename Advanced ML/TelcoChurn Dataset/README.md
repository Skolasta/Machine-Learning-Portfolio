# Telco Customer Churn Prediction

Machine learning model to predict customer churn for a telecommunications company.

## Model Performance

- **Accuracy**: 76.4%
- **Recall**: 79.2%
- **F1 Score**: 64.1%
- **ROC AUC**: 77.3%

## Features

### Top 5 Most Important Features:
1. **IsMonthToMonth** (26.2%) - Month-to-month contract indicator
2. **Contract Type** (25.9%) - Contract duration
3. **FiberNoSecurity** (6.5%) - Fiber optic without online security
4. **Internet Service Type** (3.8%) - Type of internet service
5. **FirstYear** (2.8%) - Customer in first year

### Feature Engineering:
- Financial metrics (AvgMonthlyCharges, PriceIncrease, ExpectedRevenue)
- Service combinations (TotalServices, IsPremiumCustomer)
- Risk factors (HighRiskProfile, IsNewCustomer, LoyaltyScore)
- Behavioral patterns (FiberNoSecurity, SeniorAlone)

## Files

- `telco_churn_fixed.py` - Training script
- `predict.py` - Inference script
- `telco_churn_model.pkl` - Trained model (76MB)
- `model_metadata.pkl` - Model metadata
- `feature_names.pkl` - Feature names

## Usage

### Training
```bash
python telco_churn_fixed.py
```

### Prediction
```python
from predict import predict_churn

customer = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.70,
    'TotalCharges': 151.65
}

result = predict_churn(customer)
print(result)
# {'churn_prediction': 'Yes', 'churn_probability': 0.7631, 'risk_level': 'High'}
```

## Model Details

- **Algorithm**: XGBoost with scale_pos_weight
- **Class Imbalance Handling**: scale_pos_weight=2.77
- **Hyperparameter Tuning**: RandomizedSearchCV (20 iterations, 5-fold CV)
- **Preprocessing**: RobustScaler for numerical, OneHotEncoder for categorical
- **Threshold**: 0.50 (default)

## Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
```

## Business Impact

- **High Risk Customers**: Churn probability > 70%
- **Medium Risk Customers**: Churn probability 40-70%
- **Low Risk Customers**: Churn probability < 40%

Use this model to identify at-risk customers and implement retention strategies.
