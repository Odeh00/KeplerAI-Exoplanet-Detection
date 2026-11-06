import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("جاري تحميل البيانات...")
df = pd.read_csv('cumulative.csv')

print("جاري تنظيف البيانات ومعالجة القيم المفقودة...")
cols_to_drop = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_teq_err1', 'koi_teq_err2', 'koi_pdisposition']
data = df.drop(columns=cols_to_drop, errors='ignore')

target_mapping = {'FALSE POSITIVE': 0, 'CONFIRMED': 1, 'CANDIDATE': 2}
data['koi_disposition'] = data['koi_disposition'].map(target_mapping)

numeric_cols = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

if 'koi_tce_delivname' in data.columns:
    data['koi_tce_delivname'] = data['koi_tce_delivname'].fillna(data['koi_tce_delivname'].mode()[0])
    data = pd.get_dummies(data, columns=['koi_tce_delivname'], drop_first=True)

X = data.drop(columns=['koi_disposition'])
y = data['koi_disposition']

print("جاري تطبيق الـ Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("جاري تقسيم البيانات...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

np.savez('processed_kepler_data.npz', 
         X_train=X_train, y_train=y_train.values,
         X_val=X_val, y_val=y_val.values,
         X_test=X_test, y_test=y_test.values)

print("تم الانتهاء بنجاح! تم حفظ الملف باسم: processed_kepler_data.npz")
print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")
