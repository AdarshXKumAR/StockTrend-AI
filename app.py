import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

# Load the model (ensure the model file is in the same directory)
model = load_model('keras_model.h5')


@app.route('/', methods=['GET', 'POST'])
def index():
    # Ensure the 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    if request.method == 'POST':
        stock = request.form.get('stock', 'TSLA')  # Default to 'TSLA' if no stock is entered

        # Define the start and end dates for stock data
        start = dt.datetime(2010, 1, 1)
        end = dt.datetime(2019, 12, 31)

        # Download stock data
        try:
            df = yf.download(stock, start=start, end=end)

            # Check if data is fetched successfully
            if df.empty:
                return render_template(
                    'index.html',
                    error=f"No data available for stock '{stock}' in the specified date range. Please try again.",
                )

        except Exception as e:
            return render_template(
                'index.html',
                error=f"Error fetching data for stock '{stock}': {e}. Please check the stock symbol or try later.",
            )

        # Save dataset as CSV
        csv_file_path = f"static/{stock}_data.csv"
        df.to_csv(csv_file_path)

        # Descriptive Statistics
        data_desc = df.describe()

        # Exponential Moving Averages
        ma100 = df['Close'].ewm(span=100, adjust=False).mean()
        ma200 = df['Close'].ewm(span=200, adjust=False).mean()

        # Data Splitting
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        # Check if training data is empty
        if data_training.empty:
            return render_template(
                'index.html',
                error="Insufficient data for training. Please try another stock or date range.",
            )

        # Scaling Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_training_array = scaler.fit_transform(data_training)

        # Prepare Data for Prediction
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make Predictions
        y_predicted = model.predict(x_test)

        # Inverse Scaling for Predictions
        scale_factor = 1 / scaler.scale_[0]
        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Plot 1: Closing Price vs Time Chart with 100MA
        fig1, ax1 = plt.subplots(figsize=(20, 7))
        ax1.plot(df['Close'], 'y', label='Closing Price')
        ax1.plot(ma100, 'r', label='100MA')
        ax1.set_title("Closing Price vs Time Chart with 100MA")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Price")
        ax1.legend()
        ema_chart_path = "static/ema_100.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)

        # Plot 2: Closing Price vs Time Chart with 100 & 200MA
        fig2, ax2 = plt.subplots(figsize=(20, 7))
        ax2.plot(df['Close'], 'y', label='Closing Price')
        ax2.plot(ma100, 'r', label='MA 100')
        ax2.plot(ma200, 'g', label='MA 200')
        ax2.set_title("Closing Price vs Time Chart with 100MA & 200MA")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Price")
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)

        # Plot 3: Prediction vs Original Trend
        fig3, ax3 = plt.subplots(figsize=(20, 7))
        ax3.plot(y_test, 'g', label="Original Price")
        ax3.plot(y_predicted, 'r', label="Predicted Price")
        ax3.set_title("Prediction vs Original Trend")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()
        prediction_chart_path = "static/stock_prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)

        # Return the rendered template with charts and dataset
        return render_template(
            'index.html',
            plot_path_ema_100=ema_chart_path,
            plot_path_ema_100_200=ema_chart_path_100_200,
            plot_path_prediction=prediction_chart_path,
            data_desc=data_desc.to_html(classes='table table-bordered'),
            dataset_link=csv_file_path,
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
