import json

# with open("data_qwen/layer_Qwen1.5-RM-72B_pku_benchmark_answers.json", "r") as f:
#     rewards = json.load(f)

# cleaned_safety = []
# for item in rewards:
#     dict = {}
#     win_score = item["score_a"] if item["safer_response_id"] == 0 else item["score_b"]
#     lose_score = item["score_a"] if item["safer_response_id"] == 1 else item["score_b"]
#     for idx in range(len(win_score)):
#         dict[f"layer_{idx}"] = [win_score[f"layer_{idx}"], lose_score[f"layer_{idx}"]]
#     cleaned_safety.append(dict)
# json.dump(cleaned_safety, open("data_qwen/cleaned_safety.json", "w"), indent=2)

# with open("data_qwen/layer_Qwen1.5-RM-72B_preference_benchmark.json", "r") as f:
#     rewards = json.load(f)

# cleaned_preference = []
# for item in rewards:
#     dict = {}
#     assert item["winner"] in ["model_a", "model_b"]
#     win_score = item["score_a"] if item["winner"] == "model_a" else item["score_b"]
#     lose_score = item["score_a"] if item["winner"] == "model_b" else item["score_b"]
#     for idx in range(len(win_score)):
#         dict[f"layer_{idx}"] = [win_score[f"layer_{idx}"], lose_score[f"layer_{idx}"]]
#     cleaned_preference.append(dict)
# json.dump(cleaned_preference, open("data_qwen/cleaned_preference.json", "w"), indent=2)

# with open("data_qwen/layer_Qwen1.5-RM-72B_truthful_qa_benchmark.json", "r") as f:
#     rewards = json.load(f)

# cleaned_truthful = []
# for item in rewards:
#     dict = {}
#     win_score = item["score_a"]
#     lose_score = item["score_b"]
#     best_score = item["score_c"]
#     for idx in range(len(win_score)):
#         dict[f"layer_{idx}"] = [win_score[f"layer_{idx}"], lose_score[f"layer_{idx}"], best_score[f"layer_{idx}"]]
#     cleaned_truthful.append(dict)
# json.dump(cleaned_truthful, open("data_qwen/cleaned_truthful.json", "w"), indent=2)