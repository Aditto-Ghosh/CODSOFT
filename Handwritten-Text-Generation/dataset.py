import pandas as pd
train_data = pd.read_parquet("train.parquet")
test_data = pd.read_parquet("test.parquet")
validation_data = pd.read_parquet("validation.parquet")
all_text = ""
def extract_text(dataframe):

    global all_text

    for i in range(len(dataframe)):

        text = str(dataframe.iloc[i]["text"])

        all_text += text + "\n"

extract_text(train_data)
extract_text(test_data)
extract_text(validation_data)
with open("dataset.txt", "w", encoding="utf-8") as file:

    file.write(all_text)

print("dataset.txt created successfully")