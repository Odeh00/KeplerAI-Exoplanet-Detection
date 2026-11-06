import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

file_path = None
for dirname, _, filenames in os.walk('/kaggle/input'):
  for filename in filenames:
    if filename == 'cumulative.csv':
      file_path = os.path.join(dirname, filename)

df = pd.read_csv(file_path)
print('File found and successfully read from path:', file_path)

cols_to_drop = [
    'rowid',
    'kepid',
    'kepoi_name',
    'kepler_name',
    'koi_teq_err1',
    'koi_teq_err2',
    'koi_pdisposition',
]
data = df.drop(columns=cols_to_drop, errors='ignore')

data['koi_disposition'] = data['koi_disposition'].map(
    {'FALSE POSITIVE': 0, 'CONFIRMED': 1, 'CANDIDATE': 2}
)

data = data.fillna(data.median(numeric_only=True))

print('--- Preprocessing Completed Successfully ---')
print('Final Dataset Shape:', data.shape)
print('Remaining Missing Values:', data.isnull().sum().sum())
