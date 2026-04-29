import pandas as pd

df = pd.read_csv("movie.csv")
df.columns = df.columns.str.strip()
print("Columns:", df.columns)
df = df.rename(columns={'plot': 'Description', 'genre': 'Genre'})

print("\nOriginal Genre Distribution:")
print(df['Genre'].value_counts())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df['Description'],
    df['Genre'],
    test_size=0.2,
    random_state=42,
    stratify=df['Genre'])
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_vec, y_train)

from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test_vec)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nTry Your Own Input ")

while True:
    user_input = input("\nEnter movie plot (or type 'exit'): ")

    if user_input.lower() == 'exit':
        break

    user_vec = tfidf.transform([user_input])
    prediction = model.predict(user_vec)

    print("Predicted Genre:", prediction[0])
    import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb")) 
