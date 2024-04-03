import json

with open("/data/tianhao/reward_bootstrap/dataset_reg/verbose/verbose_rewards.json", "r") as f:
    rewards = json.load(f)

num_layers = len(rewards[0])
num_responses = 2

combined_rewards = {}
for i in range(num_layers):
    combined_rewards[f"layer_{i}"] = []
    for r in rewards:
        assert len(r[f"layer_{i}"]) % num_responses == 0
        combined_rewards[f"layer_{i}"].extend(r[f"layer_{i}"])

total_samples = len(combined_rewards["layer_0"])

new_ls = []

for reward_dict in [combined_rewards]:
    length = len(reward_dict["layer_0"])
    num_layers = len(reward_dict)
    assert length % num_responses == 0
    ls_new_dict = []
    for i in range(0, length, num_responses):
        ls_new_dict.append({f"layer_{j}": reward_dict[f"layer_{j}"][i : i + num_responses] for j in range(num_layers)})
    new_ls.extend(ls_new_dict)
json.dump(new_ls, open("/data/tianhao/reward_bootstrap/dataset_reg/verbose/rewards_cleaned.json", "w"), indent=2)