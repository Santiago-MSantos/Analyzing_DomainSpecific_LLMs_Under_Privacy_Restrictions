#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
from probability_utils import calculate_pz_threshold
def plot_suffix_ranking(
    prefix,
    ranking,
    *,
    log_scale=True,
    max_items=None
):
    """
    Plot horizontal bar chart of suffix probabilities for a given prefix.

    Parameters
    ----------
    prefix : str
        The prefix used to generate the ranking.
    ranking : list[dict]
        List of entries with keys: decoded, probability, match.
    log_scale : bool
        Whether to use logarithmic x-axis.
    max_items : int or None
        Limit number of suffixes shown (top-k).
    """

    if max_items is not None:
        ranking = ranking[:max_items]

    words = [entry["decoded"].strip() for entry in ranking]
    probs = [entry["probability"] for entry in ranking]

    indices = np.arange(len(probs))
    colors = ["red" if entry.get("match", False) else "blue"
              for entry in ranking]

    plt.figure(figsize=(12, 0.4 * len(words)))

    plt.barh(indices, probs, color=colors)
    ax = plt.gca()
    if log_scale:
        plt.xscale("log")


    plt.ylabel("Suffix")
    plt.title(f"Suffix probabilities for prefix: '{prefix}'\n(red = canary)")

    plt.yticks(indices, words)
    plt.gca().invert_yaxis()  

    plt.tight_layout()
    plt.show()


def plot_extractable_curves(ranking):
    n_values = [100, 1000, 10_000, 100_000]
    p = 0.9

    match_sequences = [sequence for sequence in ranking if sequence.get("match", False)]

    count_extractable_sequences = []
    for n in n_values:
        pz_threshold = calculate_pz_threshold(n, p)
        extractable_match_sequences = [sequence for sequence in match_sequences if sequence.get("probability",0)>=pz_threshold]
        count_extractable_sequences.append(len(extractable_match_sequences))

    plt.figure()
    plt.plot(n_values, count_extractable_sequences, marker="o")
    plt.xscale("log")
    plt.xlabel("Number of samples (n)")
    plt.ylabel("Number of extractable matching sequences")
    plt.title("Extractable sequences vs number of samples (p = 0.9)")
    plt.grid(True)
    plt.show()


