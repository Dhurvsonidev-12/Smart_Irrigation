import pandas as pd

weather_df = pd.read_csv("jorethang_weather_5yrs.csv")
print(weather_df.head())

crops = ["maize", "paddy", "potato", "tomato", "wheat", "ginger"]


crop_dfs = []
for crop in crops:
    temp_df = weather_df.copy()
    temp_df["crop"] = crop
    crop_dfs.append(temp_df)

merged_df = pd.concat(crop_dfs, ignore_index=True)


merged_df.to_csv("weather_crop_dataset.csv", index=False)
