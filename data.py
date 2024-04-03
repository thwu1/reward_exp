# from os import environ
# from os.path import abspath, dirname, join


# ROOT_DIR = dirname(abspath(__file__))
# PARENT_DIR = abspath(join(ROOT_DIR, ".."))
# CACHE_DIR = join(PARENT_DIR, "cache")

# environ["HF_DATASETS_CACHE"] = CACHE_DIR
# environ["TRANSFORMERS_CACHE"] = CACHE_DIR
# environ["PYTHONUNBUFFERED"] = "true"
from model.reward import get_reward_model, split_to_list, convert_to_yi_format, normalize_dict, calculate_distance
from datasets import load_dataset
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import json

def _get_split(path: str, split: str, percentage: int, dataset_cache_dir=None):
    if split == "train":
        dataset = load_dataset(
            path,
            split=f"train[:{'-2000' if percentage == 100 else str(percentage) + '%'}]",
            cache_dir=dataset_cache_dir
        )
    elif split =="pretrain":
        dataset = load_dataset(path, split="train[-2000:-1000]", cache_dir=dataset_cache_dir)
    elif split == "val":
        dataset = load_dataset(path, split="train[-1000:]", cache_dir=dataset_cache_dir)
    elif split == "debug_train":
        dataset = load_dataset(path, split="train[:640]", cache_dir=dataset_cache_dir)   
    elif split == "debug_val":  
        dataset = load_dataset(path, split="train[-640:]", cache_dir=dataset_cache_dir)   

    return dataset

def _format_sample(model_path: str):
    if "Llama" in model_path:
        def sampler(sample):
            out = []
            for answer in sample["answers"]:
                prompt = sample["prompt"].replace("\n\nHuman: ", "</s><s>[INST] ").replace("\n\nAssistant: ", " [/INST] ")[4:]
                out.append(f'{prompt}{answer["answer"]} </s>')
            return {"formatted_answers": out}

    elif "Yi" in model_path:
        def sampler(sample):
            out = []
            for answer in sample["answers"]:
                prompt = sample["prompt"].replace("\n\nHuman: ", "<|im_end|>\n<|im_start|>user\n").replace("\n\nAssistant: ", "<|im_end|>\n<|im_start|>assistant\n").replace("<|endoftext|>", "<|eos|>").replace("<|startoftext|>", "<|sos|>")[11:]
                answer = answer["answer"].replace("<|endoftext|>", "<|eos|>").replace("<|startoftext|>", "<|sos|>")
                out.append(f'{prompt}{answer}<|im_end|>') #maybe should be a newline here, consider this if training from scratch again
            return {"formatted_answers": out}
    elif "Qwen" in model_path:
        def sampler(sample):
            out = []
            for answer in sample["answers"]:
                prompt = sample["prompt"].replace("\n\nHuman: ", "<|im_end|>\n<|im_start|>user\n").replace("\n\nAssistant: ", "<|im_end|>\n<|im_start|>assistant\n").replace("<|endoftext|>", "<|eos|>").replace("<|startoftext|>", "<|sos|>")[11:]
                answer = answer["answer"].replace("<|endoftext|>", "<|eos|>").replace("<|startoftext|>", "<|sos|>")
                out.append(f'{prompt}{answer}<|im_end|>\n')
            return {"formatted_answers": out}
    else:
        raise NotImplementedError("Formatting has not been implemented for this model type, please add it here.")

    return sampler

def load_multiwise_dataset(path, split, tokenizer, no_cache: bool, model_name: str, max_length=2000, percentage=100, dataset_cache_dir=None) -> Dataset:

    use_cache = not no_cache
    data = _get_split(path, split, percentage, dataset_cache_dir=dataset_cache_dir)
    formatted = data.map(_format_sample(model_name), num_proc=32, load_from_cache_file=use_cache, desc=f"Formatting {split} split to {model_name} format.")
    print("Example formatted inputs: ", formatted[3]['formatted_answers'])
    tokenized = formatted.map(lambda p: tokenizer(
                    p,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                ), input_columns="formatted_answers", num_proc=32, load_from_cache_file=use_cache, desc=f"Tokenizing {split} split").remove_columns(['prompt', 'answers', 'turns', 'num_responses', 'source'])

    return tokenized.with_format('torch')

class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.vstack([item for f in data for item in f["input_ids"]])
        batch["attention_mask"] = torch.vstack([item for f in data for item in f["attention_mask"]])
        batch["labels"] = torch.ones((len(data), 7, 7))
        return batch
    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.truncation_side = "left"
val_dataset = load_multiwise_dataset(
            "evanfrick/random_pre", 'val', tokenizer, False, "meta-llama/Llama-2-7b-chat-hf", max_length=2000
        )
print(len(val_dataset[0]["formatted_answers"]))

print(len(val_dataset))

reward_model = get_reward_model("Nexusflow/Starling-RM-7B-regularized")

rewards = []
data = val_dataset

idx = 0
while idx < len(data):
    batch_str = data[idx : idx + 2]["formatted_answers"]
    idx += len(batch_str)
    st = []
    for i in range(len(batch_str)):
        st += batch_str[i]
    batch_str = st
    print(batch_str)
    r = reward_model.get_reward(batch_str)
    rewards.append(r)
    if idx % 20 == 0:
        json.dump(rewards, open("pretrain_rewards.json", "w"))

json.dump(rewards, open("pretrain_rewards.json", "w"))