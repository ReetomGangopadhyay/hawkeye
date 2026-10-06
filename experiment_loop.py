import itertools
import subprocess


CPUS = [1, 2, 4, 8]
GPUS = [1]
BATCH_SIZES = [32, 64, 128, 256]
WORKERS = [1, 2, 4, 8]


for cpus, gpus, batch, workers in itertools.product(
    CPUS,
    GPUS,
    BATCH_SIZES,
    WORKERS,
):
    command = (
        f"python train.py "
        f"--batch-size {batch} "
        f"--workers {workers}"
    )

    subprocess.run(
        [
            "python",
            "controller.py",
            "--cpus", str(cpus),
            "--gpus", str(gpus),
            "--batch-size", str(batch),
            "--workers", str(workers),
            "--command", command,
        ],
        check=True,
    )
