import torch
from torch import nn
from torch.nn import functional as F
import json
import random
import time

torch.set_default_dtype(torch.float32)


def get_batch(dict):
    num_layers = len(dict)
    batch = torch.tensor([dict[f"layer_{i}"] for i in range(num_layers)])
    batch = batch.transpose(0, 1)
    return batch


def get_multi_batch(ls):
    return torch.cat([get_batch(d) for d in ls], dim=0)


class LinearModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.input_batch_norm = nn.BatchNorm1d(num_layers)
        # self.input_batch_norm.requires_grad_(False)
        self.layer = nn.Linear(num_layers, 1, bias=False)
        # self.layer.weight.data.fill_(0)
        # self.layer.weight.data[-1] = 1

    def forward(self, x):
        x = x.to("cuda")
        return self.layer(self.input_batch_norm(x))

    def hinge_loss(self, pred, eps=0.05, alpha=0.0001, num_responses=3):
        if num_responses == 3:
            pred = pred.reshape(-1, 3)
            best_wrong = pred[:, 2] - pred[:, 1]
            best_correct = pred[:, 2] - pred[:, 0]
            correct_wrong = pred[:, 0] - pred[:, 1]
            loss = -(torch.clamp(best_wrong, max=eps) + torch.clamp(best_correct, max=eps) + torch.clamp(correct_wrong, max=eps)).mean() / 3
        elif num_responses == 2:
            pred = pred.reshape(-1, 2)
            correct_wrong = pred[:, 0] - pred[:, 1]
            loss = -torch.clamp(correct_wrong, max=eps).mean()
        else:
            raise ValueError("num_responses must be 2 or 3")
        # penalty l1 norm
        loss += alpha * torch.norm(self.layer.weight * self.input_batch_norm.weight, 1)
        return loss

    def loss(self, pred, temp=1.0, alpha=0.01, num_responses=2):
        pred = pred.reshape(-1, num_responses)
        if num_responses == 3:
            best_wrong = pred[:, 2] - pred[:, 1]
            best_correct = pred[:, 2] - pred[:, 0]
            correct_wrong = pred[:, 0] - pred[:, 1]
            loss = (-F.sigmoid(best_wrong / temp).mean() - F.sigmoid(best_correct / temp).mean() - F.sigmoid(correct_wrong / temp).mean()) / 3
            # loss = -(torch.clamp(best_wrong, max=eps) + torch.clamp(best_correct, max=eps) + torch.clamp(correct_wrong, max=eps)).mean() / 3
        elif num_responses == 2:
            correct_wrong = pred[:, 0] - pred[:, 1]
            loss = -F.sigmoid(correct_wrong).mean()
            # loss = -torch.clamp(correct_wrong, max=eps).mean()
        elif num_responses == 7:
            bs = pred.shape[0]
            loss = 0
            for j in range(7):
                log_denominator = torch.logsumexp(pred[:, j:], dim=1)
                loss += -(pred[:, j] - log_denominator)
            loss = sum(loss) / bs / (num_responses - 1)
        else:
            raise ValueError("num_responses must be 2 or 3 OR 7")
        # penalty l1 norm
        loss += alpha * torch.norm(self.layer.weight * self.input_batch_norm.weight, 1)
        return loss

    @torch.no_grad()
    def test_loss(self, x, num_responses=2):
        self.eval()
        pred = self.forward(x)
        self.train()
        return self.loss(pred, num_responses=num_responses)

    @torch.no_grad()
    def test_accu(self, x, num_responses=2):
        self.eval()

        pred = self.forward(x)
        if num_responses == 3:
            pred = pred.reshape(-1, 3)
            best_wrong = sum(pred[:, 2] - pred[:, 1] > 0)
            best_correct = sum(pred[:, 2] - pred[:, 0] >= 0)
            correct_wrong = sum(pred[:, 0] - pred[:, 1] > 0)
            acc_rate = (best_wrong + best_correct + correct_wrong) / (3 * len(pred))
        elif num_responses == 2:
            pred = pred.reshape(-1, 2)
            correct_wrong = sum(pred[:, 0] - pred[:, 1] > 0)
            acc_rate = correct_wrong / len(pred)
        elif num_responses == 7:
            pred = pred.reshape(-1, 7)
            acc_rate = sum([sum(pred[:, i] > pred[:, i + 1]) for i in range(6)]) / (len(pred) * 6)
        else:
            raise ValueError("num_responses must be 2 or 3 or 7")
        self.train()
        return acc_rate

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


if __name__ == "__main__":
    with open("/data/tianhao/reward_bootstrap/rewards_cleaned.json") as f:
        rewards = json.load(f)
    
    num_responses=3

    # shuffle the rewards
    random.seed(0)
    random.shuffle(rewards)
    print(rewards[0])

    train = rewards[: int(len(rewards) * 0.05)]
    test = rewards[int(len(rewards) * 0.05) :]
    train = get_multi_batch(train)
    test = get_multi_batch(test)



    epochs = 20000

    model = LinearModel(61).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0001)
    train_loss = []
    test_loss = []
    test_accu = []
    print("Start training, test loss: ", model.test_loss(test, num_responses=num_responses), "test accu: ", model.test_accu(test, num_responses=num_responses))
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(train)
        loss = model.loss(pred, num_responses=num_responses)
        loss.backward()
        optimizer.step()

        train_loss.append(loss)
        test_loss.append(model.test_loss(test, num_responses=num_responses))
        test_accu.append(model.test_accu(test, num_responses=num_responses))
        print(f"Epoch {epoch}: test loss: {model.test_loss(test, num_responses=num_responses)}, test accu: {model.test_accu(test, num_responses=num_responses)}")
        # time.wait(0.1)
        # time.sleep(0.1)
        if epoch % 1000 == 0:
            model.save(f"linear_model_ckpt_{epoch}.pth")

    model.save("linear_model.pth")
    print(f"Accuracy: {model.test_accu(test, num_responses=num_responses)}")
    print(f"Loss: {model.test_loss(test, num_responses=num_responses)}")
    new_model = LinearModel(61)
    new_model.load("linear_model.pth")
    new_model.to("cuda")
    print(f"Accuracy: {new_model.test_accu(test, num_responses=num_responses)}")
    print(f"Loss: {new_model.test_loss(test, num_responses=num_responses)}")
