import pandas as pd
import joblib

train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

drop_cols = [
    "Unnamed: 0", "trans_num", "first", "last",
    "street", "city", "state", "dob", "trans_date_trans_time"
]

train_data = train_data.drop(columns=drop_cols, errors='ignore')
test_data = test_data.drop(columns=drop_cols, errors='ignore')
train_data["distance"] = ((train_data["lat"] - train_data["merch_lat"])**2 + 
                          (train_data["long"] - train_data["merch_long"])**2)**0.5
test_data["distance"] = ((test_data["lat"] - test_data["merch_lat"])**2 + 
                         (test_data["long"] - test_data["merch_long"])**2)**0.5
features = [
    "amt", "lat", "long", "city_pop",
    "unix_time", "merch_lat", "merch_long",
    "distance"
]

X_train = train_data[features]
y_train = train_data["is_fraud"]

X_test = test_data[features]
y_test = test_data["is_fraud"]

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)

model.fit(X_train, y_train)
joblib.dump(model, "model.pkl")
print("Model saved successfully!")
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



