import os
import torch
from torch import nn
from transformers import AutoTokenizer, LlamaPreTrainedModel, LlamaModel
import math
import pandas as pd
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from tqdm import tqdm

## Define the reward model function class


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.PAD_ID = 0
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def set_format_fn(self, format_fn):
        self.format_fn = format_fn

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        return_hidden_states=False,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states
        rewards = {}

        sliced_hidden_states = {}

        for layer, hidden_state in enumerate(hidden_states):
            scores = []
            sliced_h = []
            if layer != len(hidden_states) - 1:
                hidden = self.model.norm(hidden_state)
            else:
                hidden = hidden_state
            r = self.v_head(hidden).squeeze(-1)
            bs = int(input_ids.shape[0])
            for i in range(bs):
                c_inds = (input_ids[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
                scores.append(r[i, c_ind - 1])
                sliced_h.append(hidden_states[layer][i, c_ind - 1, :])
                # print(f"{layer}", r[i, c_ind - 1], self.v_head(self.model.norm(hidden_states[layer][i, c_ind - 1, :])) if layer != len(hidden_states) - 1 else self.v_head(hidden_states[layer][i, c_ind - 1, :]))
            scores = torch.stack(scores).detach()
            sliced_h = torch.stack(sliced_h).detach().to("cpu")
            rewards[f"layer_{layer}"] = scores
            # print(sliced_h.shape)
            sliced_hidden_states[f"layer_{layer}"] = sliced_h
        # print(sliced_hidden_states["layer_0"].shape)

        if return_hidden_states:
            return rewards, sliced_hidden_states
        else:
            return rewards

    def prepare_input_string(self, samples):
        # samples = [split_to_list(p) for p in samples]
        samples = [self.format_fn(p) for p in samples]
        return samples

    @torch.no_grad()
    def get_hidden_state(self, samples):
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = self.tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(self.get_device())
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]
        mbs = 32
        out = None
        for i in tqdm(range(math.ceil(len(samples) / mbs)), disable=True):
            _, hidden_states = self(
                input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs], return_hidden_states=True
            )
            if not out:
                out = hidden_states
            else:
                for k, v in hidden_states.items():
                    out[k] = torch.hstack((out[k], v))
        return out

    @torch.no_grad()
    def get_reward(self, samples):
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = self.tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(self.get_device())
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]
        mbs = 16
        out = None
        for i in tqdm(range(math.ceil(len(samples) / mbs))):
            rewards = self(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
            if not out:
                out = rewards
            else:
                for k, v in rewards.items():
                    out[k] = torch.hstack((out[k], v))
        return out


## Load the model and tokenizer
def get_reward_model(name):
    assert name in [
        "berkeley-nest/Starling-RM-34B",
        "berkeley-nest/Starling-RM-7B-alpha",
        "Nexusflow/Starling-RM-7B-regularized",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    if name == "berkeley-nest/Starling-RM-34B":
        reward_model = LlamaForSequenceClassification.from_pretrained("berkeley-nest/Starling-RM-34B", torch_dtype=torch.bfloat16, device_map="auto")
        reward_tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B-Chat")
        print("reward pad token id", reward_tokenizer.pad_token_id)
        reward_tokenizer.truncation_side = "left"
        reward_model.set_tokenizer(reward_tokenizer)
        reward_model.set_format_fn(convert_to_yi_format)
    elif name == "berkeley-nest/Starling-RM-7B-alpha":
        reward_model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, device_map="auto")
        directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith("model.bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        reward_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        reward_tokenizer.pad_token = reward_tokenizer.unk_token
        reward_tokenizer.truncation_side = "left"
        reward_model.set_tokenizer(reward_tokenizer)
        reward_model.set_format_fn(convert_to_llama_format)
    elif name == "Nexusflow/Starling-RM-7B-regularized":
        print("Using regularized 7B model")
        reward_model = LlamaForSequenceClassification.from_pretrained(
            "Nexusflow/Starling-RM-7B-regularized", torch_dtype=torch.bfloat16, device_map="auto", token="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"
        )

        reward_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        reward_tokenizer.pad_token = reward_tokenizer.unk_token
        reward_tokenizer.truncation_side = "left"
        reward_model.set_tokenizer(reward_tokenizer)
        reward_model.set_format_fn(convert_to_llama_format)
    elif name == "meta-llama/Llama-2-7b-chat-hf":
        reward_model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.bfloat16, device_map="auto")
        reward_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        reward_tokenizer.pad_token = reward_tokenizer.unk_token
        reward_tokenizer.truncation_side = "left"
        reward_model.set_tokenizer(reward_tokenizer)
        reward_model.set_format_fn(convert_to_llama_format)

    reward_model.eval().requires_grad_(False)
    return reward_model


def split_to_list(prompt):
    prompt = prompt.rstrip("<|end_of_turn|>")
    conversation_list = prompt.split("<|end_of_turn|>")
    processed_list = []
    for coversation in conversation_list:
        coversation = coversation.replace("GPT4 Correct Assistant:", "")
        coversation = coversation.replace("GPT4 Correct User:", "")
        processed_list.append(coversation.strip())
    return processed_list


## Inference over test prompts with Yi chat template
def convert_to_yi_format(conversation_list):
    prompt = ""
    for i, text in enumerate(conversation_list):
        if i % 2 == 0:  # Assuming user starts the conversation
            prompt += "<|im_start|>user\n" + text.strip() + "<|im_end|>\n"
        else:  # Assistant's turn
            prompt += "<|im_start|>assistant\n" + text.strip() + "<|im_end|>\n"
    return prompt.strip("\n")


def convert_to_llama_format(conversation_list):
    prompt = ""
    for i, text in enumerate(conversation_list):
        if i % 2 == 0:  # Assuming user starts the conversation
            prompt += "[INST] " + text.strip() + " "
        else:  # Assistant's turn
            prompt += "[/INST] " + text.strip() + "</s> "
    return prompt.strip()


def normalize_dict(reward_dict):
    normalized_dict = {}
    for key, value in reward_dict.items():
        normalized_dict[key] = (value - value.mean()) / value.std()
    return normalized_dict


def calculate_distance(reward_dict):
    num_keys = len(reward_dict.keys())
    dis = torch.zeros(num_keys, num_keys)
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            dis[i][j] = torch.dist(reward_dict[f"layer_{i}"], reward_dict[f"layer_{j}"])
            dis[j][i] = dis[i][j]
    plt.figure(figsize=(10, 10))
    plt.imshow(dis, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig("reward_distance.png")
    plt.close()


# model = get_reward_model("Nexusflow/Starling-RM-7B-regularized")
# samples = pd.read_csv("test_samples.csv")
# # ref_rewards = torch.tensor(samples["Reward"].tolist())
# samples = [p + r for p, r in zip(samples["Prompt"], samples["Response"])]
# samples = [split_to_list(p) for p in samples]
# samples = [convert_to_yi_format(p) for p in samples]

# reward_for_test_sample = get_reward(samples)
# normalized_rewards = normalize_dict(reward_for_test_sample)
# print(normalized_rewards)
# calculate_distance(normalized_rewards)
