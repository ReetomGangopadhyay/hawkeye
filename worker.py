import argparse
import csv
import socket
import subprocess
import time
from pathlib import Path

import psutil


def get_gpu_info():
    """
    Collect static information about the first visible GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().splitlines()

        if not lines:
            raise RuntimeError("No GPUs returned by nvidia-smi")

        name, driver, memory = [
            x.strip() for x in lines[0].split(",")
        ]

        return {
            "gpu_name": name,
            "gpu_driver": driver,
            "gpu_memory_total_mb": float(memory),
        }

    except Exception:
        return {
            "gpu_name": "unknown",
            "gpu_driver": "unknown",
            "gpu_memory_total_mb": 0,
        }


def get_gpu_stats():
    """
    Collect current utilization information from all visible GPUs.
    """
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
            util, used, total = [
                float(x.strip())
                for x in line.split(",")
            ]

            rows.append(
                {
                    "gpu_util": util,
                    "gpu_mem_used": used,
                    "gpu_mem_total": total,
                }
            )

        if not rows:
            return None

        return {
            "gpu_util": (
                sum(x["gpu_util"] for x in rows)
                / len(rows)
            ),
            "gpu_mem_used": sum(
                x["gpu_mem_used"] for x in rows
            ),
            "gpu_mem_total": sum(
                x["gpu_mem_total"] for x in rows
            ),
        }

    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def run(
    command,
    output,
    cpus,
    gpus,
    batch_size,
    workers,
):
    """
    Run one workload and collect performance telemetry.
    """

    hostname = socket.gethostname()
    gpu_info = get_gpu_info()

    print()
    print("System")
    print("-----------------------")
    print(f"Hostname:       {hostname}")
    print(f"GPU:            {gpu_info['gpu_name']}")
    print(f"Driver:         {gpu_info['gpu_driver']}")
    print(
        f"GPU memory:     "
        f"{gpu_info['gpu_memory_total_mb']} MB"
    )

    print()
    print("Experiment")
    print("-----------------------")
    print(f"CPUs:           {cpus}")
    print(f"GPUs:           {gpus}")
    print(f"Batch size:     {batch_size}")
    print(f"Workers:        {workers}")
    print(f"Command:        {command}")
    print()

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

        # interval=1 means this loop samples approximately
        # once per second.
        cpu_samples.append(
            psutil.cpu_percent(interval=1)
        )

        memory_samples.append(
            psutil.virtual_memory().percent
        )

        gpu = get_gpu_stats()

        if gpu is not None:

            gpu_util_samples.append(
                gpu["gpu_util"]
            )

            if gpu["gpu_mem_total"] > 0:
                gpu_memory_samples.append(
                    100
                    * gpu["gpu_mem_used"]
                    / gpu["gpu_mem_total"]
                )

    runtime = time.time() - start

    avg_cpu = (
        sum(cpu_samples) / len(cpu_samples)
        if cpu_samples
        else 0
    )

    avg_memory = (
        sum(memory_samples) / len(memory_samples)
        if memory_samples
        else 0
    )

    avg_gpu = (
        sum(gpu_util_samples) / len(gpu_util_samples)
        if gpu_util_samples
        else 0
    )

    avg_gpu_memory = (
        sum(gpu_memory_samples) / len(gpu_memory_samples)
        if gpu_memory_samples
        else 0
    )

    result = {
        "hostname": hostname,

        "gpu_name": gpu_info["gpu_name"],
        "gpu_driver": gpu_info["gpu_driver"],
        "gpu_memory_total_mb": gpu_info[
            "gpu_memory_total_mb"
        ],

        "cpus": cpus,
        "gpus": gpus,
        "batch_size": batch_size,
        "workers": workers,

        "runtime": runtime,

        "cpu_util": avg_cpu,
        "memory_util": avg_memory,

        "gpu_util": avg_gpu,
        "gpu_memory_util": avg_gpu_memory,

        "exit_code": process.returncode,
    }

    output = Path(output)

    # Make sure the parent directory exists.
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys(),
        )

        writer.writeheader()
        writer.writerow(result)

    print()
    print("Results")
    print("-----------------------")

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--command",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--cpus",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    result = run(
        command=args.command,
        output=args.output,
        cpus=args.cpus,
        gpus=args.gpus,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    # Propagate workload failure to SGE.
    raise SystemExit(result["exit_code"])


if __name__ == "__main__":
    main()