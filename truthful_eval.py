from model.reward import get_reward_model, split_to_list, convert_to_yi_format, normalize_dict, calculate_distance
import pandas as pd
from datasets import load_dataset
import json

# data = load_dataset("truthful_qa", "generation")["validation"]
# # randomly subsample to 100, first shuffle
# data = data.shuffle(seed=42)

# data = data.select(range(100))

with open("/data/tianhao/reward_bootstrap/dataset_old/truthful/truthful_qa_benchmark.json", "r") as f:
    data = json.load(f)
rewards = []

reward_model = get_reward_model("Nexusflow/Starling-RM-7B-regularized")

idx = 0
while idx < len(data):
    batch_str = data[idx : idx + 5]
    idx += len(batch_str)
    st = []
    for i in range(len(batch_str)):
        st.append([batch_str[i]["prompt"], batch_str[i]["response_a"]])
        st.append([batch_str[i]["prompt"], batch_str[i]["response_b"]])
        st.append([batch_str[i]["prompt"], batch_str[i]["response_c"]])
    batch_str = reward_model.prepare_input_string(st)
    print(batch_str)
    r = reward_model.get_reward(batch_str)
    # print(r)
    rewards.append(r)
    if idx % 100 == 0:
        json.dump(rewards, open("truthful_rewards.json", "w"))

json.dump(rewards, open("truthful_rewards.json", "w"))
