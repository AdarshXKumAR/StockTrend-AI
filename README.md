# StockTrend-AI

A Flask web application for stock price prediction using LSTM neural networks. Features interactive visualizations with moving averages, comparative trend analysis, and downloadable datasets. Built with Keras, pandas, and matplotlib for financial market analysis.

![Stock Prediction Demo](https://github.com/AdarshXKumAR/StockTrend-AI/blob/main/demo.png)

## ✨ Features

- **Stock Data Analysis**: Analyze historical stock data using yfinance API
- **Machine Learning Predictions**: LSTM neural network model for predicting future stock prices
- **Interactive Visualizations**: Display moving averages (100MA, 200MA) and prediction vs actual trends
- **Descriptive Statistics**: View key statistical data for each stock
- **Dataset Export**: Download analyzed stock data as CSV files

## 🛠️ Technologies Used

- **Backend**: Flask, Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Keras, TensorFlow, scikit-learn
- **Data Visualization**: Matplotlib
- **Frontend**: HTML, CSS, Bootstrap
- **Financial Data**: Yahoo Finance API (yfinance)

## 🔃 Installation

1. Clone the repository:
   ```
   git clone https://github.com/AdarshXKumAR/StockTrend-AI.git
   cd StockTrend-AI
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Make sure you have the pre-trained Keras model file:
   - Place your `keras_model.h5` file in the root directory
   - If you don't have one, you will need to train your own LSTM model

## Usage

1. Run the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Enter a stock ticker (e.g., AAPL, MSFT, TSLA) and submit to see the analysis and predictions

## 📚 Application Structure

```
StockTrend-AI/
│
├── app.py                  # Main Flask application
├── keras_model.h5          # Pre-trained LSTM model
├── requirements.txt        # Python dependencies
├── static/                 # Generated charts and CSV files
└── templates/
    └── index.html          # Web interface
```

## Model Information

The application uses a Long Short-Term Memory (LSTM) neural network pre-trained on historical stock data. The model:

- Uses 100 days of historical data to predict the next day's closing price
- Is trained on 70% of the available historical data
- Applies MinMaxScaler to normalize data before prediction
- Returns the predicted trend compared to actual stock prices

## Future Improvements

- Add more technical indicators (RSI, MACD, Bollinger Bands)
- Implement backtesting functionality to evaluate model performance
- Create portfolio analysis with multiple stocks
- Add user accounts to save favorite stocks and analyses
- Incorporate sentiment analysis from financial news

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by Yahoo Finance API
- Inspired by stock market technical analysis techniques
- Built with modern Python data science libraries

---

**Note**: This application is for educational purposes only and should not be used for actual financial investment decisions.
