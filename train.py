import argparse
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    """
    Synthetic classification dataset.

    cpu_work controls how much CPU-side preprocessing each
    sample requires.
    """

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

        # Generate synthetic input
        x = torch.randn(self.input_dim)

        # Artificial CPU preprocessing
        for _ in range(self.cpu_work):
            x = torch.sin(x) + torch.cos(x)

        y = torch.randint(
            0,
            self.num_classes,
            (1,),
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


def train(args):

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Make sure the qsub job requested a GPU."
        )

    device = torch.device("cuda")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()
    print("Configuration")
    print("-----------------------")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers:    {args.workers}")
    print(f"CPU work:   {args.cpu_work}")
    print(f"Steps:      {args.steps}")
    print()

    dataset = SyntheticDataset(
        cpu_work=args.cpu_work,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = SimpleModel().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    start = time.time()

    step = 0

    for x, y in loader:

        if step >= args.steps:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        output = model(x)

        loss = criterion(output, y)

        loss.backward()

        optimizer.step()

        if step % 20 == 0:
            print(
                f"step={step:4d} "
                f"loss={loss.item():.4f}"
            )

        step += 1

    # Wait for GPU operations to finish
    torch.cuda.synchronize()

    runtime = time.time() - start

    samples = step * args.batch_size

    print()
    print("Results")
    print("-----------------------")
    print(f"Runtime:    {runtime:.2f} sec")
    print(f"Samples:    {samples}")
    print(f"Throughput: {samples / runtime:.2f} samples/sec")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--cpu-work",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    train(args)
