"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedMedian, FedTrimmedAvg, Krum, Strategy

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Set up the chosen aggregation strategy and run federated training.

    The `aggregation` run-config selects which strategy defends the global
    model against whatever attack the malicious clients are using
    (see `client_app.py`):

    - `fedavg` — plain federated averaging, no robustness (baseline).
    - `median` — coordinate-wise median of all updates; outliers pull the
      median much less than they'd pull a mean.
    - `trimmed` — coordinate-wise trimmed mean; discards the highest/lowest
      `beta` fraction of values per coordinate before averaging the rest.
    - `krum` — picks the single client update that is most "central"
      (closest to its neighbours), and discards the rest; needs an
      estimate of how many malicious clients to tolerate.
    """

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    #pick aggregation method
    aggregation: str = context.run_config["aggregation"]
    num_malicious: int = context.run_config["num-malicious"]

    strategy: Strategy
    if aggregation == "fedavg":
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)
    elif aggregation == "median":
        strategy = FedMedian(fraction_evaluate=fraction_evaluate)
    elif aggregation == "trimmed":
        strategy = FedTrimmedAvg(fraction_evaluate=fraction_evaluate, beta=0.2)
    elif aggregation == "krum":
        strategy = Krum(fraction_evaluate=fraction_evaluate, num_malicious_nodes=num_malicious)
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}' use (fedavg/median/trimmed/krum)")

    print(f"\n>>> Defense : {aggregation}   |   Malicious CLients : {num_malicious}\n")

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the aggregated global model on the server's own held-out
    CIFAR-10 test set (not any single client's data). Called by the
    strategy once per round after aggregation, so its accuracy reflects
    how well the current defence is protecting the global model from
    whatever poisoning the malicious clients are attempting.
    """

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
