"""Accuracy-vs-round curves for the scale attack, comparing fedavg and krum."""

import csv

import matplotlib.pyplot as plt

AGGREGATIONS = ["fedavg", "krum"]
ATTACK = "scale"
RANDOM_CHANCE = 0.10


def load_curve(path, aggregation, attack):
    rows = [row for row in csv.DictReader(open(path)) if row["aggregation"] == aggregation and row["attack"] == attack]
    rows.sort(key=lambda row: int(row["round"]))
    rounds = [int(row["round"]) for row in rows]
    accuracy = [float(row["accuracy"]) for row in rows]
    return rounds, accuracy


fig, ax = plt.subplots(figsize=(8, 5))

for aggregation in AGGREGATIONS:
    rounds, accuracy = load_curve("training_curves.csv", aggregation, ATTACK)
    ax.plot(rounds, accuracy, marker="o", label=aggregation)

ax.axhline(RANDOM_CHANCE, linestyle="--", color="gray", linewidth=1)
ax.text(0, RANDOM_CHANCE, "random chance", va="bottom", ha="left", color="gray")

ax.set_xlabel("Round")
ax.set_ylabel("Accuracy")
ax.set_title("Update-scaling attack: FedAvg collapses, Krum holds\n4 of 10 clients malicious")
ax.legend(title="Aggregation")
ax.set_ylim(0, 1)

fig.tight_layout()
fig.savefig("collapse_curve.png", dpi=150)
