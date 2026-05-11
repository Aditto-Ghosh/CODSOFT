from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

data = pd.read_csv("Churn_Modelling.csv")

data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

data = pd.get_dummies(data, drop_first=True)

X = data.drop("Exited", axis=1)
y = data["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    credit_score = int(request.form["credit_score"])
    age = int(request.form["age"])
    tenure = int(request.form["tenure"])
    balance = float(request.form["balance"])
    products = int(request.form["products"])
    salary = float(request.form["salary"])

    geography_germany = 0
    geography_spain = 0

    gender_male = 0

    prediction_data = [[
        credit_score,
        age,
        tenure,
        balance,
        products,
        1,
        1,
        salary,
        geography_germany,
        geography_spain,
        gender_male
    ]]

    prediction = model.predict(prediction_data)

    if prediction[0] == 1:
        result = "Customer May Leave"
    else:
        result = "Customer Will Stay"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)