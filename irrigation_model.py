import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import time

# 1️⃣ Load dataset
df = pd.read_csv("irrigation_data.csv")
print("\n--- Dataset Info ---")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(df.head())

# 2️⃣ Drop NaNs & duplicates
df = df.dropna(subset=['pump'])
df = df.drop_duplicates()
df['pump'] = df['pump'].apply(lambda x: 1 if x >= 1 else 0)

# 3️⃣ Encode categorical features (crop)
# le_crop = LabelEncoder()
# df['crop'] = le_crop.fit_transform(df['crop'])

# 4️⃣ Features & target
X = df[['soil_moisture','temperature','humidity','rainfall']]
y = df['pump']

# 5️⃣ Split into train/test with stratify to keep class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--Class distribution before SMOTE--")
print(y_train.value_counts())
print("Percentage Distribution:\n", y_train.value_counts(normalize=True)*100)

# 6️⃣ Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n--Class distribution after SMOTE--")
print(y_train_res.value_counts())

# 7️⃣ Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 8️⃣ Train XGBoost with scale_pos_weight option removed (SMOTE handles imbalance)
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

print("\nTraining model...")
start = time.time()
model.fit(X_train_scaled, y_train_res)
end = time.time()
print(f"Training completed in {end-start:.2f} seconds")

# 9️⃣ Evaluate on real-world (imbalanced) test set
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:,1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Irrigation', 'Irrigation']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

#  🔟 Save model, scaler, and label encoder
joblib.dump(model, "irrigation_model_xgb_balanced.pkl")
joblib.dump(scaler, "scaler.pkl")
# joblib.dump(le_crop, "labelencoder_crop.pkl")
print("\n✅ Model, scaler, and encoder saved successfully!")
