import json
import torch
import argparse
from linear import LinearModel
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_responses", type=int, default=2)
argparser.add_argument("--category", type=str, default="preference")
argparser.add_argument("--mode", type=str, default="fail")
args = argparser.parse_args()

with open(f"/data/tianhao/reward_bootstrap/dataset_reg/{args.category}/rewards_cleaned.json", "r") as f:
    rewards = json.load(f)
# with open(f"/data/tianhao/reward_bootstrap/rewards_cleaned.json", "r") as f:
#     rewards = json.load(f)

num_responses = args.num_responses
print(len(rewards))

layers = len(rewards[0])
for layer in range(layers):
    correct = 0
    total = 0
    for r_dict in rewards:
        r = r_dict[f"layer_{layer}"]
        if num_responses == 3:
            correct += float(r[0] > r[1]) + float(r[2] >= r[0]) + float(r[2] > r[1])
            total += 3
        elif num_responses == 2:
            correct += float(r[0] > r[1])
            total += 1
        elif num_responses == 7:
            for i in range(6):
                correct += float(r[i] > r[i + 1])
            total += 6
    print(f"Layer {layer} test accuracy: {correct / total}")


def early_exit_aggregate(r1, r2, min=35, aggregate_num=10, can_equal=False):
    assert len(r1) == len(r2)
    if can_equal:
        judges = [r1[i] >= r2[i] for i in range(len(r1))]
        # find consecutive aggregate_num True
        for i in range(min, len(judges) - aggregate_num + 1):
            if sum(judges[i : i + aggregate_num]) == aggregate_num:
                return True
            elif sum(judges[i : i + aggregate_num]) == 0:
                return False
        else:
            return r1[-1] >= r2[-1]
    else:
        judges = [r1[i] > r2[i] for i in range(len(r1))]
        # find consecutive aggregate_num True
        for i in range(min, len(judges) - aggregate_num + 1):
            if sum(judges[i : i + aggregate_num]) == aggregate_num:
                return True
            elif sum(judges[i : i + aggregate_num]) == 0:
                return False
        else:
            return r1[-1] > r2[-1]


def JS_divergence(logp_1, logp_2):
    # logp_1 and logp_2 are log probabilities
    p = torch.exp(logp_1)
    q = torch.exp(logp_2)
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))


def contrastive_aggregate(r1, r2, min=20, eps=0, alpha=0.4, can_equal=False):
    assert len(r1) == len(r2)
    # take logsumexp
    logits = torch.tensor([r1, r2])
    logprob = (logits - torch.logsumexp(logits, dim=0)).transpose(0, 1)

    largest_JS = 0
    largest_JS_idx = 0
    for i in range(min, len(logprob)):
        js = JS_divergence(logprob[i], logprob[-1])
        if js > largest_JS:
            largest_JS = js
            largest_JS_idx = i

    # print(largest_JS)
    contrastive_logits = logprob[-1] - alpha * logprob[largest_JS_idx] if largest_JS > eps else logprob[-1]
    if can_equal:
        return contrastive_logits[0] >= contrastive_logits[1]
    else:
        return contrastive_logits[0] > contrastive_logits[1]


def majority_vote(r1, r2, min=57, can_equal=False):
    assert len(r1) == len(r2)
    if can_equal:
        judges = [r1[i] >= r2[i] for i in range(min, len(r1))]
        if sum(judges) >= len(judges) / 2:
            return True
        else:
            return False
    else:
        judges = [r1[i] > r2[i] for i in range(min, len(r1))]
        if sum(judges) >= len(judges) / 2:
            return True
        else:
            return False


# criteria = [early_exit_aggregate, majority_vote, contrastive_aggregate]
        
def to_prob(r0,r1):
    r0 = torch.tensor(r0)
    r1 = torch.tensor(r1)
    # print(r0)
    logits = torch.stack([r0, r1])
    prob = torch.softmax(logits, dim=0)
    return prob[0]

def plot_prob(r0,r1, mode="fail"):
    p = to_prob(r0,r1)
    if mode == "fail":
        if p[-1] < 0.5:
            plt.plot(p)
    else:
        plt.plot(p)

plt.figure()
mode = args.mode
for r_dict in rewards[120:130]:
    if num_responses == 3:
        r0 = [r_dict[f"layer_{layer}"][0] for layer in range(len(r_dict))]
        r1 = [r_dict[f"layer_{layer}"][1] for layer in range(len(r_dict))]
        r2 = [r_dict[f"layer_{layer}"][2] for layer in range(len(r_dict))]
        
        plot_prob(r0,r1, mode)
        plot_prob(r2,r1, mode)
        plot_prob(r2,r0, mode)
    elif num_responses == 2:
        r0 = [r_dict[f"layer_{layer}"][0] for layer in range(len(r_dict))]
        r1 = [r_dict[f"layer_{layer}"][1] for layer in range(len(r_dict))]
        plot_prob(r0,r1, mode)

plt.savefig("prob.png")