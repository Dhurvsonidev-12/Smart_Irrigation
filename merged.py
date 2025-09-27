import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv('weather_crop_dataset.csv')
df2 = pd.read_csv('irrigation_crop_data_50k.csv')

base_moisture = 50
factor = 0.5

df1['soil_moisture'] = base_moisture + df1['rainfall'] - (df1['temp_max'] - df1['temp_min'])*factor
df1['soil_moisture'] = df1['soil_moisture'].clip(0,100)

df1['pump'] = np.where(df1['soil_moisture']<40,1,0)

df1['temperature'] = (df1['temp_max']-df1['temp_min'])

df1 = df1[['temperature', 'rainfall', 'crop', 'soil_moisture', 'pump']]

df2 = df2[['temperature', 'rainfall', 'crop', 'soil_moisture', 'pump']]

merged_df = pd.concat([df1,df2],ignore_index=True)

# --- Step 5: Merge datasets ---
merged_df = pd.concat([df1, df2], ignore_index=True)

# --- Step 6: Encode categorical feature 'crop' ---
le = LabelEncoder()
merged_df['crop'] = le.fit_transform(merged_df['crop'])

merged_df = merged_df.round(2)

# --- Optional: Save merged dataset ---
merged_df.to_csv('merged_irrigation_dataset.csv', index=False)



print("Merged dataset ready for ML training!")
print(merged_df.head())