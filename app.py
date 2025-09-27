from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from datetime import timedelta, datetime
from dotenv import load_dotenv
import joblib
import pandas as pd
import requests
import random
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.permanent_session_lifetime = timedelta(days=5)

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY")
app.secret_key = SECRET_KEY

import os

models_path = os.path.join(os.path.dirname(__file__), "models")
model_file = os.path.join(models_path, "irrigation_model_xgb_balanced.pkl")
scaler_file = os.path.join(models_path, "scaler.pkl")

# Check if files exist
print("Model exists:", os.path.exists(model_file))
print("Scaler exists:", os.path.exists(scaler_file))

# Now load them only if they exist
if os.path.exists(model_file) and os.path.exists(scaler_file):
    import joblib
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
else:
    raise FileNotFoundError("Model or scaler file not found in 'models/' folder!")


# Load main training dataset for crops list
df = pd.read_csv("irrigation_crop_data_50k.csv")
crops = df['crop'].unique().tolist()
le_crop = LabelEncoder()
le_crop.fit(crops)

# ------------------- Load crop reference dataset -------------------
crop_ref = pd.read_csv("crop_data.csv")  # must contain: crop,kc,duration_days,region
crop_map = dict(zip(crop_ref['crop'].str.lower(), crop_ref['kc']))
region_map = dict(zip(crop_ref['crop'].str.lower(), crop_ref['region']))

# ------------------- Weather API -------------------
API_KEY = os.getenv("WEATHER_API_KEY")  # set in .env
LAT = "27.10696"
LON = "88.32332"

# ------------------- Simulated sensor data -------------------
sensor_data = {
    "soil_moisture": 42,
    "temperature": 30,
    "reservoir": 58,
    "liters": 1200,
    "pump": False,
    "mode": "auto",
    "current_mode": "auto"
}

# ------------------- Dummy users -------------------
users = {"Dhruv": "shubh", "Alok": "123alok"}

# ------------------- Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("dashboard.html")  # ✅ new landing page

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            error = "Username already exists!"
        else:
            users[username] = password  # ✅ save new user
            session['user'] = username
            return redirect(url_for('dashboard'))
    return render_template('signup.html', error=error)

@app.route('/login', methods=['GET', 'POST'])



def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('dashboard'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)




@app.route('/dashboard')
def dashboard(): 
    if 'user' in session: 
        temperature, humidity, rainfall = get_weather() 
        return render_template( 
            'dashboard.html', 
            username=session['user'], 
            crops=crops,
              # humidity=humidity,
               )
    else: 
        return redirect(url_for('login')
              )

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ------------------- Weather -------------------
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 0)
        return temperature, humidity, rainfall
    else:
        print("Weather API error:", data)
        return None, None, None

# ------------------- API Endpoints -------------------
@app.route('/api/get_sensor_data')
def get_sensor_data():
    # ✅ Always fetch live weather
    temperature, humidity, rainfall = get_weather()

    if sensor_data['mode'] == 'auto':
        sensor_data['pump'] = sensor_data['reservoir'] > 10

    return jsonify({
        "soil_moisture": sensor_data['soil_moisture'],
        "temperature": temperature,   # ✅ live temp
        "humidity": humidity,         # ✅ live humidity
        "rainfall": rainfall,         # ✅ send rainfall too
        "reservoir": sensor_data['reservoir'],
        "liters": sensor_data['liters'],
        "pump": sensor_data['pump'],
        "mode": sensor_data['mode'],
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    mode = data.get('mode')
    if mode in ['auto', 'manual']:
        sensor_data['mode'] = mode
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid mode"}), 400

@app.route('/api/control_pump', methods=['POST'])
def control_pump():
    if sensor_data['mode'] != 'manual':
        return jsonify({"status": "error", "message": "Pump can only be controlled in manual mode"}), 403
    data = request.get_json()
    pump_state = data.get('pump')
    if isinstance(pump_state, bool):
        sensor_data['pump'] = pump_state
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid pump state"}), 400

@app.route('/api/add_sensor_data', methods=['POST'])
def add_sensor_data():
    data = request.get_json()
    if 'soil_moisture' in data:
        sensor_data['soil_moisture'] = data['soil_moisture']
    if 'temperature' in data:
        sensor_data['temperature'] = data['temperature']
    return jsonify({"status": "success"})

@app.route('/api/predict', methods=['POST'])
def predict():
    soil_moisture = sensor_data['soil_moisture']
    temperature, humidity, rainfall = get_weather()

    humidity = int(humidity) if humidity is not None else 50

    
    if temperature is None:
        return jsonify({"error": "Weather API failed"}), 500

    data_input = request.get_json()
    crop_name = data_input.get("crop", "wheat")
    area = float(data_input.get("area", 1))

    # Encode crop for ML
    # crop_encoded = le_crop.transform([crop_name])[0]

    features_order = ['soil_moisture','temperature','humidity','rainfall']

    features = pd.DataFrame([{
    "soil_moisture": soil_moisture,
     "humidity": humidity,
    "temperature": temperature,
    "rainfall": rainfall
    }])[features_order]

    features_scaled = scaler.transform(features)
    irrigation_needed = int(model.predict(features_scaled)[0])

    # ---------------- Water Requirement ----------------
    kc = crop_map.get(crop_name.lower(), 1.0)
    region = region_map.get(crop_name.lower(), "unknown")

    # Adjust reference ET0 based on region
    eto = 4.5 if "hilly" in region else 5.5

    # Correct daily water requirement (liters/day)
    etc = kc * eto  # mm/day
    water_needed = round(area * etc, 2)  # liters/day

    log = {
        "timestamp": datetime.now(),
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "rainfall": rainfall,
        "crop": crop_name,
        "area": area,
        "region": region,
        "irrigation_needed": irrigation_needed,
        "water_needed_liters": float(water_needed),
    }

    df_log = pd.DataFrame([log])
    df_log.to_csv("realtime_logs.csv", mode="a",
                  header=not os.path.exists("realtime_logs.csv"),
                  index=False)

    return jsonify(log)

# ------------------- Simulation -------------------
@app.route('/simulate')
def simulate():
    sensor_data['soil_moisture'] = max(0, min(100, sensor_data['soil_moisture'] + random.randint(-5, 5)))
    sensor_data['temperature'] = max(0, min(50, sensor_data['temperature'] + random.randint(-2, 2)))
    sensor_data['reservoir'] = max(0, min(100, sensor_data['reservoir'] + random.randint(-3, 3)))
    sensor_data['liters'] = max(0, sensor_data['liters'] + random.randint(-50, 50))
    return jsonify(sensor_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port or 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)