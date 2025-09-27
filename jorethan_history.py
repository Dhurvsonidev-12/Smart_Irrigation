import requests
import pandas as pd
from tqdm import tqdm

lat, lon = 27.18, 88.27
start_year, end_year = 2015, 2024

all_data = []

for year in tqdm(range(start_year, end_year + 1), desc="Fetching years"):
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,relative_humidity_2m_min,soil_moisture_0_7cm"
        f"&timezone=Asia/Kolkata"
    )

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily = data["daily"]

        df = pd.DataFrame({
            "date": daily["time"],
            "temperature": [(max_temp + min_temp)/2 for max_temp, min_temp in zip(daily["temperature_2m_max"], daily["temperature_2m_min"])],
            "humidity": [(max_h + min_h)/2 for max_h, min_h in zip(daily["relative_humidity_2m_max"], daily["relative_humidity_2m_min"])],
            "soil_moisture": daily["soil_moisture_0_7cm"]
        })

        # Smart pump logic
        def pump_logic(row):
            score = 0
            if row["soil_moisture"] < 0.25:
                score += 2
            if row["temperature"] > 30:
                score += 1
            if row["humidity"] < 50:
                score += 1
            return 1 if score >= 2 else 0

        df["pump"] = df.apply(pump_logic, axis=1)

        all_data.append(df)
    else:
        print(f"❌ Failed for {year}: {response.status_code}")

# Combine all years
full_df = pd.concat(all_data).reset_index(drop=True)

# Save to CSV
full_df.to_csv("jorethang_daily_weather_with_smart_pump.csv", index=False)
print("✅ Saved 10 years of DAILY weather + smart pump ON/OFF to CSV")
