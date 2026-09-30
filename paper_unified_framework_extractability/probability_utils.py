#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from itertools import islice
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
from enum import Enum
from Levenshtein import distance as distance_levenshtein
from Levenshtein import jaro_winkler
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

P_PARAMETER = 0.9 #n-p discoverable extraction p parameter.
TEXT_COLUMN = "TEXT"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng()
samples_max_length = 5
context_len = 3
random.seed(42)

sentence_encoder = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

class Distance(str, Enum):
    EXACT = "exact"
    SUPERSEQUENCE = "supersequence"
    APPROXIMATE = "approximate" #Also called Levenshtein distance
    BLEU = "bleu"
    COSINE = "cosine" #Cosine similarity
    JAROWINKLER = "jaro-winkler"
    SUBSEQUENCE = "subsequence"



    
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


def sorting_ranking(secrets_probabilities, secrets_paired_to_context, tokenizer, distance, distance_threshold, allow_filler_tokens=False):
    standardized = {tuple(k): v for k, v in secrets_probabilities.items()}
    ranking = sorted(standardized.items(), key=lambda x: x[1], reverse=True)  #Highest probability first
    correct_secrets = secrets_paired_to_context
    enriched_ranking = []
    correct_secrets_embeddings = None
    if distance == Distance.COSINE:
        correct_secrets_embeddings = [sentence_encoder.encode(s) for s in correct_secrets]
    for phrase, probability in ranking:
        match = False
        phrase = "".join(phrase)
        if allow_filler_tokens:
            correct_secret = find_matching_secret(phrase, correct_secrets, distance, distance_threshold=0.9, correct_secrets_embeddings=correct_secrets_embeddings)
            if correct_secret != None:
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
    topk=False,
    allow_filler_tokens=False,
    rank_among_R=True,
    k=1000,
    depth=5,
    temperature = 1,
    distance_metric : Distance = "supersequence", #Allows filler tokens. Should be renamed to similarity (1 = Match)
    distance_threshold = 0.9 #For approximate distance and BLEU. Min value to match.
    ):
    
    """Computing Rényi divergence (default model: OLMo-3) of the suffix ranking and the amount of attempts to extract secrets_paired_to_context given context"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prediction_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"
    ).to(DEVICE)
    distance_metric = Distance(distance_metric)
    if checkpoint_dir != "":
        prediction_model = load_model(prediction_model, checkpoint_dir, dp)

    prediction_model.eval()
    tokenized_context = tokenizer.encode(
        context,
        add_special_tokens=False
    )
    context_len = len(tokenized_context)
    generated_sequences = []
    if topk:
        secrets_prob, suffix_probs_list, sequence_count, generated_sequences = calculate_topk_probabilities(tokenizer, 
                                                                                        prediction_model,
                                                                                          tokenized_context, 
                                                                                          sequences, 
                                                                                          distance=distance_metric,
                                                                                          distance_threshold=distance_threshold,
                                                                                          allow_filler_tokens=allow_filler_tokens,
                                                                                          rank_among_R=rank_among_R, k=k, depth=depth,
                                                                                          temperature=temperature)
    else: #Legacy, without mask over termination token.
        secrets_prob, suffix_probs_list, sequence_count = calculate_greedy_probabilities(tokenizer,
                                                                                           prediction_model, 
                                                                                           tokenized_context, 
                                                                                           sequences, 
                                                                                           temperature=temperature)
        generated_sequences = secrets_prob

    print(f"Total sequence count: {len(suffix_probs_list)}")

    ranking = sorting_ranking(
        secrets_prob,
        secrets_paired_to_context,
        tokenizer,
        distance_metric,
        distance_threshold,
        allow_filler_tokens=allow_filler_tokens
    )

    if sequence_count == 0:
        print("No secrets found for the specified key.")
        return None, None, None, None


    divergence = calculate_divergence(suffix_probs_list, sequence_count, KL=KL, alpha=alpha)


    return divergence, ranking, suffix_probs_list, generated_sequences



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


def parse_sorted_sequences(sorted_sequences, tokenizer, tokenized_context):
    parsed_sequences = []
    for seq, logprob in sorted_sequences:
        seq_without_prefix = seq[len(tokenized_context):]
        sequence_string = tokenizer.decode(seq_without_prefix, add_special_tokens=False)
        parsed_sequences.append({
            "decoded": sequence_string,
            "prob": get_probability(logprob)
        })
    return parsed_sequences



def should_continue(allow_filler_tokens, rank_among_R, has_prefix_match):
    """True if we continue generating, False if we prune this branch"""
    if has_prefix_match: #If the new sequence is a prefix of any of the target sequences, keep generating
        return True
    if allow_filler_tokens: #If it is not a prefix but we allow fillers (or missing tokens), keep generating
        return True
    return (not rank_among_R)

def calculate_topk_probabilities(tokenizer, prediction_model, tokenized_context, sequences_to_eval, distance=None, 
                                 distance_threshold = None, k=1000, depth=5, max_batch_size=128, 
                                 rank_among_R=True, allow_filler_tokens=False, epsilon=0, temperature=1):
    """
    rank_among_R: If true, we calculate the probabilities of the sequences to eval masking the termination token and tokens outside top-k. If false, we calculate the probabilities of the generated sequences
    allow_filler_tokens: If false, we prune branches that are not substrings of sequences, if True, we do not prune since filler tokens are allowed.
    """
    if distance == Distance.EXACT:
        allow_filler_tokens = False
    else:
        allow_filler_tokens = True

    if tokenizer.name_or_path == "google/gemma-2-2b" or  tokenizer.name_or_path == "google/vaultgemma-1b":
        indexes_to_mask = [tokenizer.convert_tokens_to_ids("<eos>")]
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        bos_id = tokenizer.convert_tokens_to_ids("<bos>")
        indexes_to_mask.append(end_of_turn_id)
        tokenized_context = [bos_id] + tokenized_context
    else:
        indexes_to_mask = tokenizer.convert_tokens_to_ids("<eos>") #100257 in Olmo 3
    all_sequence_probs = []
    target_token_seqs = [
                        torch.tensor(tokenizer.encode(s, add_special_tokens=False), device=DEVICE)
                        for s in sequences_to_eval]
    sequence_count = 0
    secrets_prob = {}
    prediction_model.eval()
    initial_input = torch.tensor([tokenized_context], device=DEVICE)
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
            batch_logprobs = torch.tensor([logprob for _, logprob in batch_sequences], device=DEVICE)  # [batch_size]

            with torch.no_grad():
                logits = prediction_model(batch_seqs).logits[:, -1, :]  # [batch_size, vocab_size]
                logits[:, indexes_to_mask] = float("-inf")

                topk_logits, topk_indices = torch.topk(logits, k, dim=-1) # Both: [batch_size, k]
                
                topk_logprobs = torch.log_softmax(topk_logits/temperature, dim=-1)  # [batch_size, k]
 
            # Expand sequences
            for i in range(len(batch_sequences)):
                seq = batch_seqs[i]
                seq_logprob = batch_logprobs[i].item()
                for j in range(k):
                    token_id = topk_indices[i, j].item()
                    token_logprob = topk_logprobs[i, j].item()
                    new_seq = torch.cat([seq, torch.tensor([token_id], device=DEVICE)])
                    new_seq_withoutcontext = new_seq[len(tokenized_context):]
                    has_prefix_match = is_prefix_of_any(
                                        new_seq_withoutcontext,
                                        target_token_seqs
                                        )
                    if should_continue(allow_filler_tokens, rank_among_R, has_prefix_match):
                        new_sequences.append((new_seq, seq_logprob + token_logprob + epsilon))

        sequences = new_sequences

    sequence_count = 0
    sorted_sequences = sorted(sequences, key=lambda x: x[1], reverse=True)
    
  
    suffix_probabilities = calculate_suffix_probabilities(sorted_sequences, tokenizer, tokenized_context, sequences_to_eval, allow_filler_tokens, rank_among_R, distance, distance_threshold=distance_threshold)
    

    sequence_count = len(suffix_probabilities)
    probs = list(suffix_probabilities.values())
    for suf, prob in suffix_probabilities.items():
        print(f"Sequence: {suf}, Probability: {prob}")

    sorted_sequences = parse_sorted_sequences(sorted_sequences, tokenizer, tokenized_context)
        
    return suffix_probabilities, probs, sequence_count, sorted_sequences

def calculate_suffix_probabilities(sorted_sequences, tokenizer, tokenized_context, sequences_to_eval, allow_filler_tokens, rank_among_R, distance, distance_threshold=0.9):
    suffix_probabilities = defaultdict(float)
    correct_secrets_embeddings = None
    if distance == Distance.COSINE:
        correct_secrets_embeddings = [sentence_encoder.encode(s) for s in sequences_to_eval]
    if rank_among_R:
            for seq, logprob in sorted_sequences:
                seq_without_prefix = seq[len(tokenized_context):]
                sequence_string = tokenizer.decode(seq_without_prefix, add_special_tokens=False)
                found_sequence = find_matching_secret(sequence_string, sequences_to_eval, distance, distance_threshold, correct_secrets_embeddings)
                if found_sequence != None:
                    print("Generated: ",sequence_string, "similar to: ", found_sequence, "with logprob", logprob )
                    suffix_probabilities[found_sequence] += get_probability(logprob)
    else: #Calculate probabilities of secrets among generated sequences
        for seq, logprob in sorted_sequences:
            sequence_string = tokenizer.decode(seq[len(tokenized_context):], add_special_tokens=False)
            suffix_probabilities[sequence_string] += logprob

        for secret in suffix_probabilities.keys(): #Converting logprobs to probabilities
           suffix_probabilities[secret] = get_probability(suffix_probabilities[secret])

    return suffix_probabilities

def calculate_greedy_probabilities(tokenizer, prediction_model, tokenized_context, sequences, temperature=1, device=DEVICE):
    """Legacy implementation without mask over end of sequence token, replicate this behavior using top-k with k=1"""
    assert temperature > 0, f"Temperature must be positive, got {temperature}"
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
            device=DEVICE
        )

        with torch.no_grad():
            outputs = prediction_model(tokens)
            logits = outputs.logits  # (1, seq_len, vocab)

        logits = logits - logits.max(dim=-1, keepdim=True).values
        predictions_prob = F.softmax(logits/temperature, dim=-1)

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


def contains_ordered_tokens(short, long): #All string comparison functions are case-sensitive
    search_start = 0
    for char_short in short:
        found = False
        for index_long in range(search_start, len(long)):
            if char_short == long[index_long]:
                search_start = index_long + 1
                found = True
                break
        if not found:
            return False
    return True


def find_matching_secret(phrase, correct_secrets, distance, distance_threshold, correct_secrets_embeddings=None):
    if distance == Distance.SUPERSEQUENCE:
         for secret in correct_secrets:
            if contains_ordered_tokens(secret, phrase): #Re-ejecutar con >= distance_threshold
                return secret
    if distance == Distance.APPROXIMATE:
        for secret in correct_secrets:
            if levenshtein_similarity(secret, phrase) >= distance_threshold:
                return secret
    if distance == Distance.BLEU:
        for secret in correct_secrets:
            if bleu_score(secret, phrase) >= distance_threshold:
                return secret
    if distance == Distance.COSINE:
        embedding_phrase = sentence_encoder.encode(phrase)
        for index, secret_embedding in enumerate(correct_secrets_embeddings):
            if cosine_similarity(secret_embedding, embedding_phrase) >= distance_threshold:
                return correct_secrets[index]
    if distance == Distance.JAROWINKLER:
        for secret in correct_secrets:
            if jaro_winkler_similarity(secret, phrase) >= distance_threshold:
                return secret
    if distance == Distance.SUBSEQUENCE:
        for secret in correct_secrets:
            if secret in phrase:
                return secret
    if distance == Distance.EXACT:
        for secret in correct_secrets:
            if secret == phrase:
                return secret
    return None


def jaro_winkler_similarity(short, long):
    return jaro_winkler(short, long)

def cosine_similarity(short_embedding, long_embedding):
    num = np.dot(short_embedding, long_embedding)
    denom = np.linalg.norm(short_embedding) * np.linalg.norm(long_embedding)
    return num / denom if denom != 0 else 0

def levenshtein_similarity(short, long):
    distance = distance_levenshtein(short, long)
    return 1 - distance/max(len(short), len(long))

def bleu_score(short, long):
    smooth = SmoothingFunction().method1
    return sentence_bleu([short], long, smoothing_function=smooth)

# In[ ]:


def has_enough_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False)) >= samples_max_length



def generate_sequences(tokenizer, random_space_dir="NOTEEVENTS_deanonymized.csv", random_start=True, random_seed=42):
    """Generates random sequences from random_space_dir and adds the canary sequences from Dolma 3"""

    df = pd.read_csv(random_space_dir, on_bad_lines="skip")
    candidates = df[TEXT_COLUMN].dropna()
    candidates = candidates[candidates.apply(lambda x: has_enough_tokens(tokenizer, x))]
    random.seed(random_seed)
    sampled = random.sample(list(candidates), 500)
    unique_truncated = set()

    sampled_mimic = []
    for text in tqdm(sampled):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if random_start:
            random_start = rng.integers(low=0, high=len(tokens)-samples_max_length)
        else:
            random_start = 0
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

