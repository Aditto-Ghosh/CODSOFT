from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")


@app.route('/')
def home():
    return render_template('Body.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form.get('amount', 0))
        lat = float(request.form.get('lat', 0))
        long = float(request.form.get('long', 0))
        city_pop = int(request.form.get('city_pop', 0))
        unix_time = int(request.form.get('unix_time', 0))
        merch_lat = float(request.form.get('merch_lat', 0))
        merch_long = float(request.form.get('merch_long', 0))

        distance = ((lat - merch_lat)**2 + (long - merch_long)**2)**0.5

        print("INPUT:", amount, lat, long, city_pop, unix_time, merch_lat, merch_long, distance)

        input_data = np.array([[amount, lat, long, city_pop, unix_time, merch_lat, merch_long, distance]])
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        if prediction == 1:
            result = f"Fraudulent Transaction ❌ (Risk: {prob:.2f})"
        else:
            result = f"Legitimate Transaction ✅ (Risk: {prob:.2f})"

        return render_template('Body.html', prediction_text=result)

    except Exception as e:
        print("ERROR:", e)
        return render_template('Body.html', prediction_text="⚠️ Invalid Input!")

if __name__ == "__main__":
    app.run(debug=True)