# %%
import pandas as pd

label_idx_to_text = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

df = pd.read_csv("./data/text.csv", index_col=0)
df["label"] = df["label"].apply(lambda x: label_idx_to_text[int(x)])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


def df_to_jsonl(df_in):
    lines = []
    for _, row in df_in.iterrows():
        lines.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at classifying emotions from the provided text.",
                    },
                    {
                        "role": "user",
                        "content": row["text"],
                    },
                    {"role": "assistant", "content": row["label"]},
                ]
            }
        )
    return lines


# lines
train_jsonl = df_to_jsonl(train_df)
test_jsonl = df_to_jsonl(test_df)

import json


def write_jsonl_to_file(jsonl_data, file_path, first_n):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in jsonl_data[:first_n]:
            json.dump(entry, f)
            f.write("\n")


# Write the train and test JSONL data to files
write_jsonl_to_file(train_jsonl, "./data/train.jsonl", 1000)
write_jsonl_to_file(test_jsonl, "./data/test.jsonl", 1000)
