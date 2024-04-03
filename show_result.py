import json
import torch
import argparse
from linear import LinearModel
from sklearn.linear_model import LinearRegression
from pymannkendall import original_test
import random

argparser = argparse.ArgumentParser()
argparser.add_argument("--category", type=str, default="preference")
args = argparser.parse_args()

# with open(f"/data/tianhao/reward_bootstrap/dataset_reg/{args.category}/rewards_cleaned.json", "r") as f:
with open(f"/data/tianhao/reward_bootstrap/data_qwen/cleaned_{args.category}.json", "r") as f:
    rewards = json.load(f)
# with open(f"/data/tianhao/reward_bootstrap/rewards_cleaned.json", "r") as f:
#     rewards = json.load(f)

num_responses = len(rewards[0]["layer_0"])
print(f"num_responses: {num_responses}")

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


def early_exit_aggregate(r1, r2, min=16, aggregate_num=6, can_equal=False):
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


# def JS_divergence(logp_1, logp_2):
#     # logp_1 and logp_2 are log probabilities
#     return 0.5 * (torch.sum(torch.exp(logp_1) * (logp_1 - logp_2)) + torch.sum(torch.exp(logp_2) * (logp_2 - logp_1)))


def JS_divergence(logp_1, logp_2):
    # logp_1 and logp_2 are log probabilities
    p = torch.exp(logp_1)
    q = torch.exp(logp_2)
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))


def contrastive_aggregate(r1, r2, min=10, max=70, eps=0.000, alpha=1.0, can_equal=False):
    assert len(r1) == len(r2)
    # take logsumexp
    logits = torch.tensor([r1, r2])
    logprob = (logits - torch.logsumexp(logits, dim=0)).transpose(0, 1)

    largest_JS = 0
    largest_JS_idx = 0
    for i in range(min, max):
        js = JS_divergence(logprob[i], logprob[max])
        if js > largest_JS:
            largest_JS = js
            largest_JS_idx = i

    # print(largest_JS)
    contrastive_logits = logprob[max] - alpha * logprob[largest_JS_idx] if largest_JS > eps else logprob[-1]
    if can_equal:
        return contrastive_logits[0] >= contrastive_logits[1]
    else:
        return contrastive_logits[0] > contrastive_logits[1]


def mean_aggregate(r1, r2, min=10, max=32, can_equal=False):
    assert len(r1) == len(r2)
    logits = torch.tensor([r1, r2])
    prob = torch.softmax(logits, dim=0).transpose(0, 1)[:, 0]

    mean = torch.mean(prob[min:max])
    if can_equal:
        return mean >= 0.5
    else:
        return mean > 0.5


def logit_mean_aggregate(r1, r2, min=16, max=32, can_equal=False):
    assert len(r1) == len(r2)
    logits = torch.tensor([r1, r2])
    logprob = (logits - torch.logsumexp(logits, dim=0)).transpose(0, 1)[:, 0]

    mean_logprob = torch.mean(logprob[min:max]).exp()
    # print(mean_logprob.item())
    if can_equal:
        return mean_logprob >= 0.5
    else:
        return mean_logprob > 0.5


def exponential_aggregate(r1, r2, min=16, max=32, alpha=0, can_equal=False):
    assert len(r1) == len(r2)
    logits = torch.tensor([r1, r2])
    prob = torch.softmax(logits, dim=0).transpose(0, 1)[:, 0]

    prob = prob[min:max]
    exp_moving_avg = prob[0]
    for p in prob[1:]:
        exp_moving_avg = alpha * exp_moving_avg + (1 - alpha) * p
    if can_equal:
        return exp_moving_avg >= 0.5
    else:
        return exp_moving_avg > 0.5


def logit_exponential_aggregate(r1, r2, min=16, max=32, alpha=0, can_equal=False):
    assert len(r1) == len(r2)
    logits = torch.tensor([r1, r2])
    logprob = (logits - torch.logsumexp(logits, dim=0)).transpose(0, 1)[:, 0]

    logprob = logprob[min:max]
    exp_moving_avg = logprob[0]
    for p in logprob[1:]:
        exp_moving_avg = alpha * exp_moving_avg + (1 - alpha) * p
    if can_equal:
        return exp_moving_avg.exp() >= 0.5
    else:
        return exp_moving_avg.exp() > 0.5


def linear_regression_aggregate(r1, r2, min=10, max=80, can_equal=False):
    logits = torch.tensor([r1, r2])
    prob = torch.softmax(logits, dim=0).transpose(0, 1)[:, 0]

    prob = prob[min:max]
    model = LinearRegression()
    time = torch.arange(min, max).reshape(-1, 1).cpu().detach().numpy()
    model.fit(time, prob.cpu().detach().numpy().reshape(-1, 1))

    # Get the slope of the regression line
    slope = model.coef_[0][0]
    # print(slope)

    # Determine the trend based on the slope
    if can_equal:
        return slope >= 0
    else:
        return slope > 0


def majority_vote(r1, r2, min=28, can_equal=False):
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


def mann_kendall_aggregate(r1, r2, min=30, max=78, can_equal=False):
    logits = torch.tensor([r1, r2])
    prob = torch.softmax(logits, dim=0).transpose(0, 1)[:, 0]

    prob = prob[min:max].tolist()
    result = original_test(prob)
    # print(result)
    # result.slope

    trend = result.slope

    # if trend == 0 and not can_equal:
    #     trend = random.choice([-1, 1])

    # Determine the trend based on the test result
    if can_equal:
        return trend >= 0
    else:
        return trend > 0


criteria = [linear_regression_aggregate, mann_kendall_aggregate, contrastive_aggregate, early_exit_aggregate]
for cri in criteria:
    correct = 0
    total = 0
    for r_dict in rewards:
        if num_responses == 3:
            r0 = [r_dict[f"layer_{layer}"][0] for layer in range(len(r_dict))]
            r1 = [r_dict[f"layer_{layer}"][1] for layer in range(len(r_dict))]
            r2 = [r_dict[f"layer_{layer}"][2] for layer in range(len(r_dict))]

            correct += float(cri(r0, r1)) + float(cri(r2, r1)) + float(cri(r2, r0, can_equal=True))
            total += 3
        elif num_responses == 2:
            r0 = [r_dict[f"layer_{layer}"][0] for layer in range(len(r_dict))]
            r1 = [r_dict[f"layer_{layer}"][1] for layer in range(len(r_dict))]
            correct += float(cri(r0, r1))
            total += 1
    print(f"Layer {layer} early exit {str(cri)} test accuracy: {correct / total}")


# model = LinearModel(61)
# model.load("linear_model_ckpt_3000.pth")
# model = model.to("cuda")
# for idx, weight in enumerate((model.layer.weight * model.input_batch_norm.weight)[0]):
#     print(f"layer {idx} weight: {weight.item():.5f}")


# def get_batch(dict):
#     num_layers = len(dict)
#     batch = torch.tensor([dict[f"layer_{i}"] for i in range(num_layers)])
#     batch = batch.transpose(0, 1)
#     return batch


# def get_multi_batch(ls):
#     return torch.cat([get_batch(d) for d in ls], dim=0)


# test_set = get_multi_batch(rewards)
# print("Self aggregated accuracy:", model.test_accu(test_set, num_responses=num_responses).item())
