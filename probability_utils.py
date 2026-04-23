#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import random
from datasets import load_dataset

P_PARAMETER = 0.9 #n-p discoverable extraction p parameter.
TEXT_COLUMN = "TEXT"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-3-1025-7B") 
rng = np.random.default_rng()
samples_max_length = 5
context_len = 3
random.seed(42)
def compute_np_discoverable_extraction(secret_prob, p=P_PARAMETER):
    num = np.log2(1-p)
    den = np.log2(1-secret_prob)
    return num/den

def calculate_pz_threshold(n,p):
    """Required for plot of Experiment 3 (multiple prefixes). Finds the minimum p_z that is n-p discoverable extractable"""
    pz = (-(-p+1)**(1/n))+1
    return pz


def get_probability(logprob: float) -> float:
    return math.exp(logprob)


def sorting_ranking(secrets_probabilities, secrets_paired_to_context, tokenizer, allow_filler_tokens=False):
    standardized = {tuple(k): v for k, v in secrets_probabilities.items()}
    ranking = sorted(standardized.items(), key=lambda x: x[1], reverse=True)  #Highest probability first
    correct_secrets = secrets_paired_to_context
    enriched_ranking = []

    for key, probability in ranking:
        phrase = ''.join(key)
        match = False
        if allow_filler_tokens:
            correct_secret = find_matching_secret(phrase, correct_secrets)
            if correct_secret:
                match = True
                phrase = correct_secret

        else:
            match = (phrase in correct_secrets)

        attempts = compute_np_discoverable_extraction(probability)
        enriched_ranking.append({
            "decoded": phrase,
            "probability": probability,
            "attempts": attempts,
            "match": match
        })

    return enriched_ranking

def is_prefix_of_any(prefix, targets):
    prefix_len = prefix.size(0)
    return any(
        target.size(0) >= prefix_len
        and torch.equal(target[:prefix_len], prefix)
        for target in targets
    )

def load_model(model, checkpoint_dir: str, dp=False, device="cuda"):

    if dp:
        checkpoint = torch.load(str(checkpoint_dir), map_location=device)
        sanitized_checkpoint = OrderedDict()

        for k, v in checkpoint.items():
            if k.startswith("_module."):
                sanitized_checkpoint[k.replace("_module.", "")] = v
            elif k.startswith("module."):
                sanitized_checkpoint[k.replace("module.", "")] = v
            else:
                sanitized_checkpoint[k] = v

        load_info = model.load_state_dict(sanitized_checkpoint)
        print(load_info) 
        return model
    else: #loading checkpoints folder
        prediction_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype="auto"
        ).to(device)
        return prediction_model

    
def compute_renyi_divergence_alpha_and_ranking(
    sequences,
    secrets_paired_to_context,
    alpha=0.5,
    dp=False,
    context="default context ",
    KL=False,
    checkpoint_dir="",
    model_name="allenai/Olmo-3-1025-7B",
    device="cuda",
    topk=False,
    allow_filler_tokens=False,
    rank_among_R=True,
    k=1000,
    depth=5
):
    """PyTorch version of computing Rényi divergence (OLMo-3)"""


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prediction_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"
    ).to(device)

    if checkpoint_dir != "":
        prediction_model = load_model(prediction_model, checkpoint_dir, dp)

    prediction_model.eval()

    tokenized_context = tokenizer.encode(
        context,
        add_special_tokens=False
    )
    context_len = len(tokenized_context)

    if topk:
        secrets_prob, all_sequence_probs, sequence_count = calculate_topk_probabilities(tokenizer, prediction_model, tokenized_context, sequences, secrets_paired_to_context, allow_filler_tokens=allow_filler_tokens, rank_among_R=rank_among_R, k=k, depth=depth)
    else:
        secrets_prob, all_sequence_probs, sequence_count = calculate_greedy_probabilities(tokenizer, prediction_model, tokenized_context, sequences, secrets_paired_to_context)

    print(f"Total sequence count: {sequence_count}")

    ranking = sorting_ranking(
        secrets_prob,
        secrets_paired_to_context,
        tokenizer,
        allow_filler_tokens=allow_filler_tokens
    )

    if sequence_count == 0:
        print("No secrets found for the specified key.")
        return None, None, None

    divergence = calculate_divergence(all_sequence_probs, sequence_count, KL=KL, alpha=alpha)

    return divergence, ranking, all_sequence_probs



def calculate_divergence(all_sequence_probs, sequence_count, epsilon=1e-12, KL=True, alpha=0.5):
    total_prob = sum(all_sequence_probs)
    if total_prob == 0:
        return float("inf")
    normalized_probs = [(p + epsilon) / total_prob for p in all_sequence_probs]

    uniform_prob = 1.0 / sequence_count

    if KL:
        divergence = sum(
            p * np.log2(p / uniform_prob)
            for p in normalized_probs
        )
    else:
        sum_product = sum(
            (p ** alpha) * (uniform_prob ** (1 - alpha))
            for p in normalized_probs
        )
        divergence = np.log2(sum_product) / (alpha - 1)
    return divergence
def calculate_topk_probabilities(tokenizer, prediction_model, tokenized_context, sequences_to_eval, secrets_paired_to_context, device="cuda", k=1000, depth=5, max_batch_size=128, rank_among_R=True, allow_filler_tokens=False):
    """
    rank_among_R: If true, we calculate the probabilities of the sequences to eval. If false, we calculate the probabilities of the generated sequences
    allow_filler_tokens: If false, we prune branches that are not substrings of sequences, if True, we do not prune since filler tokens are allowed.
    """
    all_sequence_probs = []
    target_token_seqs = [
                        torch.tensor(tokenizer.encode(s, add_special_tokens=False), device=device)
                        for s in sequences_to_eval]
    sequence_count = 0
    secrets_prob = {}
    prediction_model.eval()
    initial_input = torch.tensor([tokenized_context], device=device)
    words_probabilities = defaultdict(int)
    # each element: (token_ids, cumulative_logprob)
    sequences = [(initial_input[0], 0.0)]

    for step in range(depth):
        print("Current depth:", step)
        new_sequences = []

        # Process sequences in batches
        for batch_start in range(0, len(sequences), max_batch_size):
            batch_end = min(batch_start + max_batch_size, len(sequences))
            batch_sequences = sequences[batch_start:batch_end]

            # Stack batch sequences
            batch_seqs = torch.stack([seq for seq, _ in batch_sequences])  # [batch_size, seq_len]
            batch_logprobs = torch.tensor([logprob for _, logprob in batch_sequences], device=device)  # [batch_size]

            with torch.no_grad():
                logits = prediction_model(batch_seqs).logits[:, -1, :]  # [batch_size, vocab_size]
                log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, vocab_size]

            # Get top-k for each sequence in the batch

            topk_logprobs, topk_indices = torch.topk(log_probs, k, dim=-1)  # Both: [batch_size, k]

            # Expand sequences
            for i in range(len(batch_sequences)):
                seq = batch_seqs[i]
                seq_logprob = batch_logprobs[i].item()
                for j in range(k):
                    token_id = topk_indices[i, j].item()
                    token_logprob = topk_logprobs[i, j].item()
                    new_seq = torch.cat([seq, torch.tensor([token_id], device=device)])
                    new_seq_withoutcontext = new_seq[len(tokenized_context):]
                    has_prefix_match = is_prefix_of_any(
                                        new_seq_withoutcontext,
                                        target_token_seqs
                                        )
                    if allow_filler_tokens or (not rank_among_R) or has_prefix_match:
                        new_sequences.append((new_seq, seq_logprob + token_logprob + 1e-12))

        sequences = new_sequences

    secrets_probabilities = defaultdict(int)
    sequence_count = 0

    sorted_sequences = sorted(sequences, key=lambda x: x[1], reverse=True)
    if rank_among_R:
        for seq, logprob in sorted_sequences:
            sequence_string = tokenizer.decode(seq, add_special_tokens=False)
            if not allow_filler_tokens:
                for eval_seq in sequences_to_eval:
                    if eval_seq in sequence_string:
                        words_probabilities[eval_seq] += get_probability(logprob)
            else:
                found_sequence = find_matching_secret(sequence_string, sequences_to_eval)
                if found_sequence:
                    print("Generated: ",sequence_string, "with logprob", logprob )
                    words_probabilities[found_sequence] += get_probability(logprob)

        print("words_probabilities:", words_probabilities)



    else: #Calculate probabilities of secrets among generated sequences
        for seq, logprob in sorted_sequences:
            sequence_string = tokenizer.decode(seq[len(tokenized_context):], add_special_tokens=False)
            words_probabilities[sequence_string] += logprob


        for words in words_probabilities.keys(): #Converting logprobs to probabilities
           words_probabilities[words] = get_probability(words_probabilities[words])

    sequence_count = len(words_probabilities)
    probs = list(words_probabilities.values())
    print(words_probabilities)
    return words_probabilities, probs, sequence_count


def calculate_greedy_probabilities(tokenizer, prediction_model, tokenized_context, sequences, secrets_paired_to_context, device="cuda"):
    all_sequence_probs = []
    sequence_count = 0
    secrets_prob = {}
    context_len = len(tokenized_context)
    for secret in sequences:
        encoded_secret = tokenizer.encode(
            secret,
            add_special_tokens=False
        )

        sequence_count += 1

        max_length = prediction_model.config.max_position_embeddings
        max_secret_len = max_length - context_len

        full_input = tokenized_context + encoded_secret[:max_secret_len]

        tokens = torch.tensor(
            [full_input],
            device=device
        )

        with torch.no_grad():
            outputs = prediction_model(tokens)
            logits = outputs.logits  # (1, seq_len, vocab)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        predictions_prob = F.softmax(logits, dim=-1)

        log_prob = 0.0
        for k in range(context_len, len(full_input)):
            token = full_input[k]
            if token == tokenizer.eos_token_id:
                break

            prob = predictions_prob[0, k - 1, token].item()
            log_prob += np.log2(prob)

        sequence_prob = 2 ** log_prob
        secrets_prob[secret] = sequence_prob
        all_sequence_probs.append(sequence_prob)

    return secrets_prob, all_sequence_probs, sequence_count

def calculate_exposure(secrets_ranking):
    space_size = len(secrets_ranking)
    avg_rank = calculate_avg_rank(secrets_ranking)
    print("The average rank of canary sequences is", avg_rank)
    return np.log2(space_size) - np.log2(avg_rank)

def calculate_avg_rank(secrets_ranking):
    """
     enriched_ranking.append({
            "decoded": phrase,
            "probability": probability,
            "match": match
        })"""
    true_match_ranks = []
    for pos in range(len(secrets_ranking)):
        if secrets_ranking[pos]["match"]==True:
            true_match_ranks.append(pos+1)

    avg_rank = np.sum(true_match_ranks) / len(true_match_ranks)
    return avg_rank



def contains_ordered_tokens(short, long): 
    short_tokens = short.split()
    long_tokens = long.split()

    it = iter(long_tokens)
    return all(token in it for token in short_tokens)


def find_matching_secret(phrase, correct_secrets):
    for secret in correct_secrets:
        if contains_ordered_tokens(secret, phrase):
            return secret

    return None


# In[ ]:


def has_enough_tokens(text):
    return len(tokenizer.encode(text, add_special_tokens=False)) >= samples_max_length



def generate_sequences(random_space_dir="NOTEEVENTS_deanonymized.csv"):
    """Generates random sequences from random_space_dir and adds the canary sequences from Dolma 3"""

    df = pd.read_csv(random_space_dir, on_bad_lines="skip")
    candidates = df[TEXT_COLUMN].dropna()
    candidates = candidates[candidates.apply(has_enough_tokens)]

    sampled = random.sample(list(candidates), 500)
    unique_truncated = set()

    sampled_mimic = []
    for text in tqdm(sampled):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        random_start = rng.integers(low=0, high=len(tokens)-samples_max_length)
        truncated_tokens = tokens[random_start:random_start+samples_max_length]
        truncated_text = tokenizer.decode(truncated_tokens)
        prev_size = len(unique_truncated)
        unique_truncated.add(truncated_text)
        if (prev_size < len(unique_truncated)):
            sampled_mimic.append(truncated_text)

    sequences = []


    dolma3 = load_dataset("allenai/dolma3_mix-6T-1025", revision="c484f1a9ceea687fe7be16b5a35dd658422ae2c1")
    for seq_idx in tqdm(range(100)):
        text = dolma3["train"][seq_idx]["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < samples_max_length+context_len:
            continue

        truncated_tokens = tokens[:samples_max_length+context_len]
        truncated_text = tokenizer.decode(truncated_tokens, add_special_tokens=False)
        sequences.append({
            "text": truncated_text,
            "source": "dolma3"
        })

    for sample in sampled_mimic:
       text = sample
       sequences.append({
        "text":text,
        "source":"MIMIC3"
        })
    return sequences

