import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_utils import load_protocol, add_audio_path
from src.features_mfcc import extract_mfcc_features

# 1. Load protocol
df_train = load_protocol(".../ASVspoof2019.LA.cm.train.trn.txt")
df_train = add_audio_path(df_train, ".../LA/train/flac")

# 2. Extract features
X_list = []
y_list = []

label_map = {"bonafide": 0, "spoof": 1}

for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
    feat = extract_mfcc_features(row["path"])
    X_list.append(feat)
    y_list.append(label_map[row["label"]])

X = np.stack(X_list)
y = np.array(y_list)
print("Feature shape:", X.shape)  # (N, D)

# 3. Train/val split (within train set)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_val = scaler.transform(X_val)

# 5. XGBoost classifier
clf = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=4
)

clf.fit(X_tr, y_tr)

from sklearn.metrics import roc_auc_score, classification_report

y_val_proba = clf.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba >= 0.5).astype(int)

print("ROC AUC:", roc_auc_score(y_val, y_val_proba))
print(classification_report(y_val, y_val_pred, target_names=["bonafide", "spoof"]))
