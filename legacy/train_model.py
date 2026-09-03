import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_model(csv_file="gold_labeled.csv", output_file="gold_predictions.csv"):
    """
    Reads gold_labeled.csv, trains an XGBoost classifier on the 80% train split,
    and generates signal probabilities on the 20% out-of-sample test set.
    """
    print(f"Loading labeled dataset from '{csv_file}'...")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    
    # Select input features and target column
    feature_cols = ['log_ret', 'ema_ratio', 'rsi', 'atr', 'volatility']
    X = df[feature_cols]
    y = df['target']
    
    # Chronological Train/Test Split (80% Train, 20% Out-of-Sample Test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)} bars")
    print(f"Out-of-sample Test set size: {len(X_test)} bars")
    
    # Initialize XGBoost Classifier with regularization settings
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,            # Shallow depth prevents overfitting
        learning_rate=0.03,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    
    print("\nTraining XGBoost model...")
    model.fit(X_train, y_train)
    
    # Generate probability scores on unseen test data
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Trigger BUY signal only when model confidence is > 55%
    confidence_threshold = 0.55
    test_preds = (test_probs > confidence_threshold).astype(int)
    
    print("\n--- Out-of-Sample Classification Report ---")
    print(classification_report(y_test, test_preds))
    
    # Save test slice with model predictions for Step 5 backtesting
    test_df = df.iloc[split_idx:].copy()
    test_df['signal_prob'] = test_probs
    test_df['signal'] = test_preds
    
    test_df.to_csv(output_file)
    print(f"Predictions saved to '{output_file}' ready for Step 5 backtesting!")
    
    return test_df

if __name__ == "__main__":
    predictions_df = train_model()