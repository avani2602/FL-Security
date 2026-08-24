"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Handle one training round for this client, and poison it if this
    client is malicious.

    Every client (honest and malicious) always trains normally with the
    same `train_fn`. The two attacks act at the two ends of that call:

    - `label_flip` poisons the *input* — `load_data(..., poison=True)`
      swaps this client's training labels before training even starts
      (see `apply_transforms_poison` in `task.py`), so training itself is
      unmodified but learns the wrong thing.
    - `scale` poisons the *output* — training proceeds normally on
      genuine data, and only the resulting weights are multiplied by 10x
      before being sent back to the server.

    Which attack (if any) applies is controlled by the `attack` run-config
    combined with whether this client's `partition_id` falls among the
    first `num-malicious` clients (see `is_malicious` below).
    """

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]

    #the first 'number-malicious' clients are the poison, everyone else is an honest client
    number_malicious: int = context.run_config["num-malicious"]
    attack: str = context.run_config["attack"]               #Label-flipping or Scaling
    # Partitions are numbered 0..num_partitions-1; treating the first
    # `number_malicious` of them as malicious is an arbitrary but fixed
    # convention, so which clients are "attackers" stays the same every round.
    is_malicious = partition_id < number_malicious

    #Label-flipping poisons the data, scaling does not this turn malicious after training
    poison = is_malicious and attack == "label_flip"
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, poison=poison)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    #build the update to send back
    if is_malicious and attack == "scale":
        # Update-scaling attack: train normally on genuine data, then
        # multiply every weight in the resulting update by 10x before
        # sending it. The goal is to let this one client's update dominate
        # (or blow up) the server's aggregation of many clients' updates.
        print(f"[client {partition_id}] is MALICIOUS - scaling it's update X10")
        scale = 10.0
        bad_state = {k: v * scale for k, v in model.state_dict().items()}
        model_record = ArrayRecord(bad_state)
    else:
        if is_malicious:
            print(f"[client{partition_id}] is MALICIOUS - flipping label")
        model_record = ArrayRecord(model.state_dict())
    
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on this client's local test data.

    This always runs on this client's genuine, un-poisoned test split
    (`load_data` only poisons the *training* split), regardless of whether
    the client is malicious, so evaluation metrics stay trustworthy.
    """

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
