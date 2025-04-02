# Stock-Market-Forecasting-Model
# Stock Market Forecasting using Deep Learning

## 📌 Overview
Stock market forecasting is a complex task due to its volatile nature. This project leverages **Long Short-Term Memory (LSTM)**, a type of Recurrent Neural Network (RNN), to predict future stock prices based on historical data.

## 🎯 Objectives
- Predict future stock prices using deep learning.
- Improve forecasting accuracy with LSTM networks.
- Deploy the model as a REST API using Flask.

## 🗂 Dataset
The dataset includes stock price data from companies like **Google (GOOGL) and Amazon (AMZN)** from **2006 to 2018**. The dataset contains the following features:
- **Date**: Timestamp of stock price.
- **Open**: Opening price.
- **High**: Highest price.
- **Low**: Lowest price.
- **Close**: Closing price.
- **Volume**: Number of shares traded.

## 🔧 Technologies Used
- **Python** (NumPy, Pandas, Matplotlib, Seaborn, Plotly)
- **Deep Learning** (TensorFlow, Keras, LSTM)
- **Flask** (For API deployment)
- **Scikit-learn** (Data preprocessing)

## 📊 Data Preprocessing
- Handling missing values
- Feature scaling using **MinMaxScaler**
- Data visualization for trend analysis

## 🤖 Model Architecture
The deep learning model is based on **LSTM** with the following architecture:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    
    LSTM(units=50),
    Dropout(0.2),
    
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
```

## 🚀 Model Training
```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
```

## 🎯 Model Evaluation
The model is evaluated using:
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

## 🌍 Deployment using Flask
### 1️⃣ Save the trained model:
```python
model.save('stock_model.h5')
```

### 2️⃣ Create a Flask API:
```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('stock_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
```

### 3️⃣ Run the Flask app:
```bash
python app.py
```

### 4️⃣ Test API with a POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"data": [some_values_here]}'
```

## 🔮 Future Enhancements
- Implement **Transformer models** for better time-series forecasting.
- Integrate **real-time news sentiment analysis** as a feature.
- Deploy the model on **AWS or Heroku** for cloud accessibility.

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Feel free to fork this repository and contribute! If you find any issues, create a pull request or open an issue.

---
### 📬 Contact
- **Author:** Uday Singh Chauhan
- **GitHub:** [your_github_username](https://github.com/uschauhan-dev)
- **LinkedIn:** [your_linkedin_profile](www.linkedin.com/in/uschauhan)

