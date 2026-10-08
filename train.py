import argparse
import json
import os
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    def __init__(
        self,
        size=100_000,
        input_dim=1024,
        num_classes=10,
        cpu_work=10,
    ):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cpu_work = cpu_work

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randn(self.input_dim)

        for _ in range(self.cpu_work):
            x = torch.sin(x) + torch.cos(x)

        y = torch.randint(
            0, self.num_classes, (1,)
        ).item()

        return x, y


class SimpleModel(nn.Module):
    def __init__(self, input_dim=1024, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def emit_event(event, **fields):
    """
    Emit structured events for worker.py.
    """
    message = {"event": event, **fields}
    print(
        "HAWKEYE_EVENT " + json.dumps(message),
        flush=True,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU unavailable. Request a compatible GPU."
        )

    device = torch.device("cuda")

    torch.set_num_threads(
        max(1, int(os.environ.get("NSLOTS", 1)))
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nConfiguration")
    print("-----------------------")
    print(f"Batch size:    {args.batch_size}")
    print(f"Workers:       {args.workers}")
    print(f"CPU work:      {args.cpu_work}")
    print(f"Warmup steps:  {args.warmup_steps}")
    print(f"Measured steps:{args.steps}")

    dataset = SyntheticDataset(
        cpu_work=args.cpu_work,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    def batches():
        while True:
            yield from loader

    iterator = batches()

    def training_step():
        x, y = next(iterator)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        return loss

    # Warmup: excluded from measured performance.
    print("\nWarmup")
    print("-----------------------")

    warmup_start = time.perf_counter()

    for step in range(args.warmup_steps):
        training_step()

    torch.cuda.synchronize()
    warmup_runtime = time.perf_counter() - warmup_start

    print(f"Warmup runtime: {warmup_runtime:.3f} s")

    # Measured training region.
    print("\nMeasured training")
    print("-----------------------")

    torch.cuda.synchronize()

    emit_event("training_start")
    start = time.perf_counter()

    for step in range(args.steps):
        loss = training_step()

        if step % 200 == 0:
            print(
                f"step={step:5d} loss={loss.item():.4f}",
                flush=True,
            )

    torch.cuda.synchronize()
    runtime = time.perf_counter() - start

    emit_event("training_end")

    samples = args.steps * args.batch_size
    throughput = samples / runtime
    step_time_ms = 1000 * runtime / args.steps

    metrics = {
        "warmup_steps": args.warmup_steps,
        "warmup_runtime": warmup_runtime,
        "training_steps": args.steps,
        "training_runtime": runtime,
        "samples": samples,
        "throughput": throughput,
        "step_time_ms": step_time_ms,
        "cpu_work": args.cpu_work,
    }

    emit_event("training_results", **metrics)

    print("\nResults")
    print("-----------------------")
    print(f"Training runtime: {runtime:.3f} s")
    print(f"Samples:          {samples}")
    print(f"Throughput:       {throughput:.2f} samples/s")
    print(f"Step time:        {step_time_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size", type=int, default=128
    )
    parser.add_argument(
        "--workers", type=int, default=4
    )
    parser.add_argument(
        "--steps", type=int, default=2000
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100
    )
    parser.add_argument(
        "--cpu-work", type=int, default=10
    )

    args = parser.parse_args()

    if args.batch_size < 1 or args.steps < 1:
        parser.error("batch-size and steps must be positive")
    if args.workers < 0 or args.warmup_steps < 0:
        parser.error("workers and warmup-steps must be nonnegative")

    train(args)


if __name__ == "__main__":
    main()
