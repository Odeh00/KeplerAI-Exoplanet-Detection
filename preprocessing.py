
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight


file_path = None

for search_dir in ['/kaggle/input', os.getcwd()]:
    if os.path.exists(search_dir):
        for dirname, _, filenames in os.walk(search_dir):
            if 'cumulative.csv' in filenames:
                file_path = os.path.join(dirname, 'cumulative.csv')
                break
        if file_path is not None:
            break

if file_path is None:
    raise FileNotFoundError("cumulative.csv was not found.")

df = pd.read_csv(file_path)

print("Dataset loaded successfully:", file_path)
print("Original Dataset Shape:", df.shape)

target_mapping = {
    'FALSE POSITIVE': 0,
    'CONFIRMED': 1,
    'CANDIDATE': 2
}

y = df['koi_disposition'].map(target_mapping)
groups = df['kepid']

valid_rows = y.notna() & groups.notna()

df = df.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].astype(int).reset_index(drop=True)
groups = groups.loc[valid_rows].reset_index(drop=True)

cols_to_drop = [
    'rowid',
    'kepid',
    'kepoi_name',
    'kepler_name',
    'koi_disposition',
    'koi_pdisposition',
    'koi_score',
    'koi_fpflag_nt',
    'koi_fpflag_ss',
    'koi_fpflag_co',
    'koi_fpflag_ec',
    'koi_teq_err1',
    'koi_teq_err2',
    'koi_tce_delivname'
]

X = df.drop(columns=cols_to_drop, errors='ignore')
X = X.dropna(axis=1, how='all')

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

transformers = [
    ('num', numeric_pipeline, numeric_cols)
]

if categorical_cols:
    transformers.append(
        ('cat', categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(
    transformers=transformers,
    sparse_threshold=0
)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_idx, temp_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_temp = X.iloc[temp_idx]

y_train = y.iloc[train_idx]
y_temp = y.iloc[temp_idx]

groups_temp = groups.iloc[temp_idx]

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

val_idx, test_idx = next(
    splitter2.split(X_temp, y_temp, groups=groups_temp)
)

X_val = X_temp.iloc[val_idx]
X_test = X_temp.iloc[test_idx]

y_val = y_temp.iloc[val_idx]
y_test = y_temp.iloc[test_idx]

X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_val_processed = preprocessor.transform(X_val).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)

y_train = y_train.to_numpy(dtype=np.int64)
y_val = y_val.to_numpy(dtype=np.int64)
y_test = y_test.to_numpy(dtype=np.int64)

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

output_dir = (
    '/kaggle/working'
    if os.path.exists('/kaggle/working')
    else os.path.join(os.getcwd(), 'preprocessing_outputs')
)

os.makedirs(output_dir, exist_ok=True)

np.savez_compressed(
    os.path.join(output_dir, 'processed_kepler_data.npz'),
    X_train=X_train_processed,
    y_train=y_train,
    X_val=X_val_processed,
    y_val=y_val,
    X_test=X_test_processed,
    y_test=y_test
)

joblib.dump(
    preprocessor,
    os.path.join(output_dir, 'kepler_preprocessor.pkl')
)

with open(os.path.join(output_dir, 'class_weights.json'), 'w') as f:
    json.dump(class_weights, f, indent=4)

train_groups = set(groups.iloc[train_idx])
val_groups = set(groups_temp.iloc[val_idx])
test_groups = set(groups_temp.iloc[test_idx])

assert len(train_groups & val_groups) == 0
assert len(train_groups & test_groups) == 0
assert len(val_groups & test_groups) == 0

assert np.isfinite(X_train_processed).all()
assert np.isfinite(X_val_processed).all()
assert np.isfinite(X_test_processed).all()

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")

print("\nDATASET SHAPES")
print("Training:", X_train_processed.shape)
print("Validation:", X_val_processed.shape)
print("Testing:", X_test_processed.shape)

print("\nCLASS WEIGHTS")
print(class_weights)

print("\nMISSING VALUES")
print("Training:", np.isnan(X_train_processed).sum())
print("Validation:", np.isnan(X_val_processed).sum())
print("Testing:", np.isnan(X_test_processed).sum())

print("\nGROUP LEAKAGE CHECK")
print("All groups are separated successfully.")

print("\nOUTPUT DIRECTORY:", output_dir)
print("All preprocessing artifacts saved successfully.")


print('--- Preprocessing Completed Successfully ---')
print('Final Dataset Shape:', data.shape)
print('Remaining Missing Values:', data.isnull().sum().sum())
