"""Grouped bar chart of final accuracy per defence, split by attack type."""

import csv

import matplotlib.pyplot as plt

DEFENCES = ["fedavg", "median", "trimmed", "krum"]
ATTACKS = ["label_flip", "scale"]
RANDOM_CHANCE = 0.10


def load_results(path):
    accuracy = {(row["aggregation"], row["attack"]): float(row["final_accuracy"]) for row in csv.DictReader(open(path))}
    return {attack: [accuracy[(defence, attack)] for defence in DEFENCES] for attack in ATTACKS}


results = load_results("results.csv")

x = range(len(DEFENCES))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

for i, attack in enumerate(ATTACKS):
    offset = (i - 0.5) * bar_width
    positions = [xi + offset for xi in x]
    ax.bar(positions, results[attack], width=bar_width, label=attack)

ax.axhline(RANDOM_CHANCE, linestyle="--", color="gray", linewidth=1)
ax.text(-0.5, RANDOM_CHANCE, "random chance", va="bottom", ha="left", color="gray")

ax.set_xticks(list(x))
ax.set_xticklabels(DEFENCES)
ax.set_xlabel("Defence (aggregation strategy)")
ax.set_ylabel("Final accuracy")
ax.set_title("Final accuracy by defence and attack type\n4 of 10 clients malicious")
ax.legend(title="Attack")
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig("results.png", dpi=150)
