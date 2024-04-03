import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time

def pl_loss(pred):
    # Implementation of Plackett-Luce loss
    # pred: (bs, responses_per_prompt)
    # We assume that the preference order is from 0 to responses_per_prompt - 1
    _, responses_per_prompt = pred.shape
    loss = 0
    for j in range(responses_per_prompt):
        log_denominator = torch.logsumexp(pred[:, j:], dim=1)
        loss += -(pred[:, j] - log_denominator)
    loss = loss.mean() / (responses_per_prompt - 1)
    return loss

def pairwise_loss(pred):
    # Implementation of pairwise loss
    # pred: (bs, responses_per_prompt)
    # We assume that the preference order is from 0 to responses_per_prompt - 1
    _, responses_per_prompt = pred.shape
    loss = 0
    for i in range(responses_per_prompt):
        for j in range(i + 1, responses_per_prompt):
            loss += torch.sigmoid(pred[:, i] - pred[:, j]).mean()
    loss /= (responses_per_prompt * (responses_per_prompt - 1) / 2)
    return -loss

def consecutive_loss(pred):
    # Implementation of consecutive loss
    # pred: (bs, responses_per_prompt)
    # We assume that the preference order is from 0 to responses_per_prompt - 1
    _, responses_per_prompt = pred.shape
    loss = 0
    for i in range(responses_per_prompt - 1):
        loss += torch.sigmoid(pred[:, i] - pred[:, i + 1]).mean()
    loss /= (responses_per_prompt - 1)
    return -loss

def consecutive_accuracy(pred):
    # Implementation of consecutive accuracy
    # pred: (bs, responses_per_prompt)
    # We assume that the preference order is from 0 to responses_per_prompt - 1
    _, responses_per_prompt = pred.shape
    acc_rate = sum(sum(pred[:, :-1] > pred[:, 1:])) / (len(pred) * (responses_per_prompt - 1))
    return acc_rate
tag = "v2"
layer = 32
train_file = [f"/data/tianhao/reward_bootstrap/hidden_states_reward_cleaned_{i}_{tag}.pt" for i in range(1)]
# test_file = [f"/data/tianhao/reward_bootstrap/hidden_states_{i}.pt" for i in range(100, 101)]
truth_file = f"/data/tianhao/reward_bootstrap/hidden_states_truthful_0_{tag}.pt"
preference_file = f"/data/tianhao/reward_bootstrap/hidden_states_preference_0_{tag}.pt"
safety_file = f"/data/tianhao/reward_bootstrap/hidden_states_safety_0_{tag}.pt"

train_dict = []
for file in train_file:
    train_dict.extend(torch.load(file))

# test_dict = []
# for file in test_file:
#     test_dict.extend(torch.load(file))

print("train dict length", len(train_dict))
# print("test dict length", len(test_dict))

train_dict_layer = [item[f"layer_{layer}"] for item in train_dict]
train_loader = DataLoader(train_dict_layer, batch_size=512, shuffle=False, drop_last=False)

# test_dict_layer = [item[f"layer_{layer}"] for item in test_dict]
# test_loader = DataLoader(test_dict_layer, batch_size=512, shuffle=False, drop_last=False)

truth_dict = torch.load(truth_file)
truth_loader = DataLoader([item[f"layer_{layer}"] for item in truth_dict], batch_size=512, shuffle=False, drop_last=False)

preference_dict = torch.load(preference_file)
preference_loader = DataLoader([item[f"layer_{layer}"] for item in preference_dict], batch_size=512, shuffle=False, drop_last=False)

safety_dict = torch.load(safety_file)
safety_loader = DataLoader([item[f"layer_{layer}"] for item in safety_dict], batch_size=512, shuffle=False, drop_last=False)

def loss_fn(pred):
    return pl_loss(pred)
    # return -pairwise_loss(pred)
    # return -consecutive_loss(pred)

def acc_fn(pred):
    return consecutive_accuracy(pred)

@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    loss_ls = []
    acc_ls = []
    for batch in test_loader:
        pred = model(batch)
        loss_ls.append(loss_fn(pred))
        acc_ls.append(acc_fn(pred))
    model.train()
    # print(loss_ls, acc_ls)
    # print("acc ls", acc_ls)
    # print("sum acc ls", sum(acc_ls))
    # print("sum loss ls", sum(loss_ls))
    return sum(loss_ls) / len(loss_ls), sum(acc_ls) / len(acc_ls)

def train(model, train_loader, test_loader, epochs=10):
    # model: nn.Module
    # train_loader: DataLoader
    # test_loader: DataLoader
    # return: None
    model.train()
    highest_acc = {"truth": 0, "preference": 0, "safety": 0}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        for batch in train_loader:
            # print(batch)
            optimizer.zero_grad()
            pred = model(batch)
            # print(pred.shape)
            loss = loss_fn(pred)
            # print(loss)
            loss.backward()
            optimizer.step()
        if test_loader is not None:
            test_loss, test_acc = evaluate(model, test_loader)
        else:
            test_loss = None
            test_acc = None
        _, truth_acc = evaluate(model, truth_loader)
        _, preference_acc = evaluate(model, preference_loader)
        _, safety_acc = evaluate(model, safety_loader)
        if truth_acc > highest_acc["truth"]:
            highest_acc["truth"] = truth_acc
        if preference_acc > highest_acc["preference"]:
            highest_acc["preference"] = preference_acc
        if safety_acc > highest_acc["safety"]:
            highest_acc["safety"] = safety_acc
        print(f"Epoch {epoch}, test loss: {test_loss}, test acc: {test_acc}, truth acc: {truth_acc}, preference acc: {preference_acc}, safety acc: {safety_acc}")

    with open("result.jsonl", "a") as f:
        highest_acc = {k: v.item() for k, v in highest_acc.items()}
        highest_acc["tag"] = tag
        highest_acc["layer"] = layer
        highest_acc["train_file"] = train_file
        highest_acc["time"] = time.time()
        print(highest_acc)
        json.dump(highest_acc, f)
        f.write("\n")


class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1, dtype=torch.bfloat16)

    def forward(self, x):
        return self.fc(x).squeeze(-1)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))

v_head = ValueHead(4096)

train(model=v_head, train_loader=train_loader, test_loader=None, epochs=1000)