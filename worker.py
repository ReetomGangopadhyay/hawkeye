import argparse
import csv
import subprocess
import time
from pathlib import Path

import psutil


def gpu_stats():
    """Return mean GPU utilization and memory usage across visible GPUs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        rows = []
        for line in result.stdout.strip().splitlines():
            util, used, total = [float(x.strip()) for x in line.split(",")]
            rows.append((util, used, total))

        if not rows:
            return None

        return {
            "gpu_util": sum(x[0] for x in rows) / len(rows),
            "gpu_mem_used": sum(x[1] for x in rows),
            "gpu_mem_total": sum(x[2] for x in rows),
        }

    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def run(command, output, cpus, gpus, batch_size, workers):
    start = time.time()

    process = subprocess.Popen(
        command,
        shell=True,
    )

    cpu_samples = []
    memory_samples = []
    gpu_util_samples = []
    gpu_memory_samples = []

    while process.poll() is None:
        cpu_samples.append(psutil.cpu_percent(interval=1))
        memory_samples.append(psutil.virtual_memory().percent)

        gpu = gpu_stats()

        if gpu:
            gpu_util_samples.append(gpu["gpu_util"])

            if gpu["gpu_mem_total"] > 0:
                gpu_memory_samples.append(
                    100 * gpu["gpu_mem_used"] / gpu["gpu_mem_total"]
                )

    runtime = time.time() - start

    result = {
        "cpus": cpus,
        "gpus": gpus,
        "batch_size": batch_size,
        "workers": workers,
        "runtime": runtime,
        "cpu_util": sum(cpu_samples) / len(cpu_samples)
        if cpu_samples else 0,
        "memory_util": sum(memory_samples) / len(memory_samples)
        if memory_samples else 0,
        "gpu_util": sum(gpu_util_samples) / len(gpu_util_samples)
        if gpu_util_samples else 0,
        "gpu_memory_util": sum(gpu_memory_samples) / len(gpu_memory_samples)
        if gpu_memory_samples else 0,
        "exit_code": process.returncode,
    }

    output = Path(output)

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)

    args = parser.parse_args()

    run(
        args.command,
        args.output,
        args.cpus,
        args.gpus,
        args.batch_size,
        args.workers,
    )