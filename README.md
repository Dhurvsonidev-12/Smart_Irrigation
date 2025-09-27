# 🌱 Smart Irrigation System with Machine Learning  

This project is an **IoT + ML-powered Smart Irrigation System** designed to help farmers and gardeners **save water, increase crop yield, and automate irrigation** decisions.  

---

## 📖 Project Overview  
Traditional irrigation systems waste water because they run on fixed schedules, without considering soil moisture, weather, or rainfall.  
Our system uses **real-time sensor data + weather API + machine learning** to intelligently decide **when and how much to irrigate**.  

✅ Reduces water wastage  
✅ Optimizes irrigation scheduling  
✅ Prevents over-watering & under-watering  
✅ Integrates **ML model** for decision-making  
✅ Provides **web dashboard (Flask backend + API)**  

---

## ⚙️ Features  
- 📡 **Real-time data collection** from soil moisture, temperature, humidity, and rainfall sensors.  
- 🌦️ **Weather forecast integration** using OpenWeather API.  
- 🤖 **Machine Learning Model (XGBoost / Random Forest)** to predict pump ON/OFF.  
- 💾 **Data storage** for historical analysis.  
- 🌍 **Web-based interface** to monitor and control irrigation remotely.  

---

## 🏗️ Tech Stack  
- **Backend:** Flask (Python)  
- **ML Models:** XGBoost, Random Forest  
- **Data Handling:** Pandas, NumPy, Scikit-learn  
- **Frontend:** HTML/CSS/JavaScript (or React if extended)  
- **Deployment:** Render / Railway / PythonAnywhere  
- **Sensors:** Soil moisture, temperature, humidity (via IoT module)  
- **External API:** OpenWeatherMap API for rainfall predictions  

---

## 📊 Dataset  
We use environmental and agricultural datasets such as:  
- [Irrigation Machine Learning Dataset](https://www.kaggle.com/datasets/gopalmahadevan/irrigation-machine-learning-dataset)  
- [Weather & Climate Data](https://www.kaggle.com/datasets/muthuj7/weather-dataset)  

---

## 🚀 How It Works  
1. Sensors send soil and weather data to the backend.  
2. Backend preprocesses the data and feeds it to the ML model.  
3. The ML model predicts whether the **pump should be ON/OFF**.  
4. Weather forecast is checked → if rain is expected, irrigation is delayed.  
5. Dashboard displays the current status, past logs, and system recommendations.  

---

## 🔮 Future Improvements  
- Add **mobile app control (Flutter/Blynk)**  
- Predict **crop-specific water needs**  
- Use **deep learning for weather-based irrigation forecasting**  
- Enable **real-time alerts (SMS/Email/Telegram)**  

---

## 📷 Project Demo  
*(Add screenshots or images of your dashboard, sensor setup, etc.)*  

---

## 👨‍💻 Contributors  
- Dhruv Soni (Project Developer)  

---