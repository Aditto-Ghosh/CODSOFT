from flask import Flask, render_template, request
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['plot']

    data = vectorizer.transform([user_input])

    result = model.predict(data)[0]

    return render_template('index.html', prediction=result)
if __name__ == "__main__":
    app.run(debug=True)
