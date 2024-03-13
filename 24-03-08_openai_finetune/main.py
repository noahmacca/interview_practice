# %%
from openai import OpenAI
import pandas as pd
import json


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
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Take subset for testing
df_train = df_train[:100]
df_test = df_test[:100]


# Set up openai helpers
client = OpenAI()


def call_openai_chat_json(sys_prompt, user_prompt, model, temperature, key):
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    try:
        return json.loads(completion.choices[0].message.content)[key]
    except Exception as e:
        print("ERROR parsing response:\n{}".format(e))
        print("RAW RESPONSE:\n{}".format(completion.choices[0].message.content))


res = call_openai_chat_json(
    "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair. Respond in json with key 'poem'.",
    "Compose a poem that explains the concept of recursion in programming.",
    "gpt-3.5-turbo-0125",
    0.15,
    "poem",
)

print(res)

# %%
# Test a few prompts and see how they do
prompts = [
    {
        "label": "baseline",
        "sys": "You are an expert at classifying emotions from the provided text. Respond in json, with the key 'class'.",
        "first_msg": """
I am an analyst who needs help at correctly classifying emotion from text. Please classify the text, highlighted in the "Input Text" heading, using ONLY the classes provided under the "Classes" heading. Output the label and no other text.

Classes
sadness
joy
love
anger
fear
surprise

Input Text
{{TEXT}}
""",
        "json_resp_key": "class",
    },
    {
        "label": "add chain of thought",
        "sys": "You are an expert at classifying emotions from the provided text. Respond with json, with keys 'thinking' and 'class'.",
        "first_msg": """
I am an analyst who needs help at correctly classifying emotion from the provided tweet text. Please classify the text, highlighted in the "Input Text" heading, using ONLY the classes provided under the "Classes" heading. Think step by step, and then output the final classification at the end of your output on a new line.

Classes
sadness
joy
love
anger
fear
surprise

Input Text
{{TEXT}}
""",
        "json_resp_key": "class",
    },
]

models = [
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "ft:gpt-3.5-turbo-0125:personal:emotions-03112024:91l6kk1Q",
    "ft:gpt-3.5-turbo-0125:personal:emotions-03112024:91le9Ouo",
    "ft:gpt-3.5-turbo-0125:personal:emotions-0311-400:91leQZRl",
]


# Call on one sample
prompt_idx = 1
model_idx = 0
prompt = prompts[prompt_idx]
test_tweet = "ive blabbed on enough for tonight im tired and ive been feeling pretty crappy from this kentucky weather"
model = models[model_idx]

res = call_openai_chat_json(
    prompt["sys"],
    prompt["first_msg"].replace("{{TEXT}}", test_tweet),
    model,
    0.0,
    prompt["json_resp_key"],
)

res


# json.loads(res)

# %%
models = [
    "gpt-3.5-turbo-0125",
    "gpt-4-0125-preview",
    "ft:gpt-3.5-turbo-0125:personal:emotions-03112024:91l6kk1Q",
    "ft:gpt-3.5-turbo-0125:personal:emotions-03112024:91le9Ouo",
    "ft:gpt-3.5-turbo-0125:personal:emotions-0311-400:91leQZRl",
]


# %%
# Call on all samples
from tqdm.auto import tqdm

dft = df_test[:100].copy()

for m in models:
    for p_idx, p in enumerate(prompts[:1]):
        tqdm.pandas(
            desc="Call promptidx={} on model={} for {} rows".format(p_idx, m, len(dft))
        )
        dft["pred_model={}_promptidx={}".format(m, p_idx)] = dft["text"].progress_apply(
            lambda x: call_openai_chat_json(
                p["sys"],
                p["first_msg"].replace("{{TEXT}}", x),
                m,
                0.0,
                p["json_resp_key"],
            )
        )

dft.head()

# %%
print(len(dft))
dft = dft.dropna()
print(len(dft))
# %%
# Evaluate performance
from sklearn.metrics import accuracy_score

scores = []

# Calculate the accuracy of the predictions
for col in [i for i in dft.columns if "pred_" in i]:
    score = (accuracy_score(dft["label"], dft[col]),)
    run = (col.replace("pred_", ""),)
    scores.append({"run": run, "score": score})
    print("run={} score={}".format(run, score))

pd.DataFrame(scores)

# %% Use the best prompt to fine-tune
prompt_idx = 0


# Write 200 lines of training data with prompt 0
def df_to_jsonl(df_in, sys_msg, first_msg):
    lines = []
    for _, row in df_in.iterrows():
        lines.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": sys_msg,
                    },
                    {
                        "role": "user",
                        "content": first_msg.replace("{{TEXT}}", row["text"]),
                    },
                    {"role": "assistant", "content": row["label"]},
                ]
            }
        )
    return lines


# lines
sys_msg = prompts[prompt_idx]["sys"]
first_msg = prompts[prompt_idx]["first_msg"]


import json


def write_jsonl_to_file(jsonl_data, file_path, first_n):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in jsonl_data[:first_n]:
            json.dump(entry, f)
            f.write("\n")


# Write the train and test JSONL data to files
train_jsonl = df_to_jsonl(df_train, sys_msg, first_msg)
test_jsonl = df_to_jsonl(df_test, sys_msg, first_msg)

write_jsonl_to_file(train_jsonl, "./data/emotions_03112024_400_train.jsonl", 400)
write_jsonl_to_file(test_jsonl, "./data/emotions_03112024_400_test.jsonl", 400)

# %%
# Upload training file
from openai import OpenAI

client.files.create(
    file=open("./data/emotions_03112024_400_train.jsonl", "rb"), purpose="fine-tune"
)
client.files.create(
    file=open("./data/emotions_03112024_400_test.jsonl", "rb"), purpose="fine-tune"
)


# %%
# Test the fine-tuned model performance on the test set
