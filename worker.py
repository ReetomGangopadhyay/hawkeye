import argparse
import csv
import json
import os
import queue
import shlex
import socket
import subprocess
import threading
import time
from pathlib import Path

import psutil


EVENT_PREFIX = "HAWKEYE_EVENT "


def get_gpu_info():
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
            timeout=10,
        )

        line = result.stdout.strip().splitlines()[0]
        name, driver, memory = [
            value.strip() for value in line.split(",")
        ]

        return {
            "gpu_name": name,
            "gpu_driver": driver,
            "gpu_memory_total_mb": float(memory),
        }

    except (OSError, ValueError, IndexError,
            subprocess.SubprocessError):
        return {
            "gpu_name": "unknown",
            "gpu_driver": "unknown",
            "gpu_memory_total_mb": 0.0,
        }


def get_gpu_stats():
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
            timeout=5,
        )

        rows = []
        for line in result.stdout.strip().splitlines():
            util, used, total = [
                float(value.strip())
                for value in line.split(",")
            ]
            rows.append((util, used, total))

        if not rows:
            return None

        return {
            "gpu_util": sum(r[0] for r in rows) / len(rows),
            "gpu_memory_used_mb": sum(r[1] for r in rows),
            "gpu_memory_total_mb": sum(r[2] for r in rows),
        }

    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def process_tree_stats(root_pid, previous):
    """
    Collect CPU time and RSS for the workload process tree.

    CPU utilization is normalized to one logical CPU:
    100% means approximately one fully utilized CPU core.
    """
    try:
        root = psutil.Process(root_pid)
        processes = [root] + root.children(recursive=True)
    except psutil.Error:
        return None, previous

    cpu_time = 0.0
    rss = 0
    identities = set()

    for proc in processes:
        try:
            with proc.oneshot():
                identity = (proc.pid, proc.create_time())
                times = proc.cpu_times()
                memory = proc.memory_info()

            identities.add(identity)
            cpu_time += times.user + times.system
            rss += memory.rss

        except psutil.Error:
            continue

    now = time.monotonic()
    cpu_percent = None

    if previous is not None:
        old_time, old_cpu, old_ids = previous
        elapsed = now - old_time

        # Tree membership can change when DataLoader workers
        # start or exit, so this is an approximate measurement.
        if elapsed > 0 and identities == old_ids:
            cpu_percent = max(
                0.0,
                100.0 * (cpu_time - old_cpu) / elapsed,
            )

    return {
        "cpu_util": cpu_percent,
        "memory_rss_mb": rss / (1024 * 1024),
    }, (now, cpu_time, identities)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def output_reader(stream, events):
    """
    Forward workload output and parse structured events.
    """
    for line in stream:
        print(line, end="", flush=True)

        if line.startswith(EVENT_PREFIX):
            try:
                event = json.loads(
                    line[len(EVENT_PREFIX):]
                )
                events.put(event)
            except json.JSONDecodeError:
                pass


def run(command, output, cpus, gpus, batch_size, workers):
    hostname = socket.gethostname()
    gpu_info = get_gpu_info()

    print("\nSystem")
    print("-----------------------")
    print(f"Hostname: {hostname}")
    print(f"GPU: {gpu_info['gpu_name']}")
    print(f"Driver: {gpu_info['gpu_driver']}")

    print("\nExperiment")
    print("-----------------------")
    print(f"CPUs: {cpus}")
    print(f"GPUs: {gpus}")
    print(f"Batch size: {batch_size}")
    print(f"Workers: {workers}")
    print(f"Command: {command}", flush=True)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    events = queue.Queue()
    telemetry = []
    training_metrics = {}

    launch_start = time.perf_counter()

    # Start the command directly, without a shell, so its
    # PID is the workload process rather than a shell PID.
    process = subprocess.Popen(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    reader = threading.Thread(
        target=output_reader,
        args=(process.stdout, events),
        daemon=True,
    )
    reader.start()

    training_active = False
    saw_start = False
    saw_end = False
    previous_cpu = None
    sample_interval = 0.25

    while process.poll() is None or not events.empty():
        try:
            event = events.get(timeout=0.05)

            if event.get("event") == "training_start":
                training_active = True
                saw_start = True
                previous_cpu = None

            elif event.get("event") == "training_end":
                training_active = False
                saw_end = True

            elif event.get("event") == "training_results":
                training_metrics.update(event)

        except queue.Empty:
            pass

        if not training_active or process.poll() is not None:
            continue

        sample_start = time.monotonic()

        proc_stats, previous_cpu = process_tree_stats(
            process.pid, previous_cpu
        )
        gpu_stats = get_gpu_stats()

        sample_end = time.monotonic()

        # Ignore samples that cross the end of training
        # when the end event is already available.
        pending = []
        while True:
            try:
                pending.append(events.get_nowait())
            except queue.Empty:
                break

        for event in pending:
            if event.get("event") == "training_end":
                training_active = False
                saw_end = True
            elif event.get("event") == "training_results":
                training_metrics.update(event)
            elif event.get("event") == "training_start":
                training_active = True
                saw_start = True

        if training_active:
            gpu_memory_percent = None

            if gpu_stats is not None:
                total = gpu_stats["gpu_memory_total_mb"]
                if total > 0:
                    gpu_memory_percent = (
                        100 * gpu_stats["gpu_memory_used_mb"] / total
                    )

            telemetry.append({
                "elapsed_since_launch_s":
                    time.perf_counter() - launch_start,
                "cpu_util":
                    proc_stats["cpu_util"]
                    if proc_stats else None,
                "memory_rss_mb":
                    proc_stats["memory_rss_mb"]
                    if proc_stats else None,
                "gpu_util":
                    gpu_stats["gpu_util"]
                    if gpu_stats else None,
                "gpu_memory_util": gpu_memory_percent,
            })

        elapsed = sample_end - sample_start
        time.sleep(max(0, sample_interval - elapsed))

    exit_code = process.wait()
    reader.join(timeout=5)

    # Drain events emitted immediately before process exit.
    while not events.empty():
        event = events.get_nowait()

        if event.get("event") == "training_start":
            saw_start = True
        elif event.get("event") == "training_end":
            saw_end = True
        elif event.get("event") == "training_results":
            training_metrics.update(event)

    worker_runtime = time.perf_counter() - launch_start

    valid_training = (
        exit_code == 0
        and saw_start
        and saw_end
        and "training_runtime" in training_metrics
    )

    result = {
        "hostname": hostname,
        "gpu_name": gpu_info["gpu_name"],
        "gpu_driver": gpu_info["gpu_driver"],
        "gpu_memory_total_mb":
            gpu_info["gpu_memory_total_mb"],

        "cpus": cpus,
        "gpus": gpus,
        "batch_size": batch_size,
        "workers": workers,

        "runtime": worker_runtime,
        "worker_runtime": worker_runtime,
        "warmup_steps": training_metrics.get("warmup_steps"),
        "warmup_runtime": training_metrics.get("warmup_runtime"),
        "training_steps": training_metrics.get("training_steps"),
        "training_runtime": training_metrics.get("training_runtime"),
        "samples": training_metrics.get("samples"),
        "throughput": training_metrics.get("throughput"),
        "step_time_ms": training_metrics.get("step_time_ms"),
        "cpu_work": training_metrics.get("cpu_work"),

        "cpu_util": mean(
            [r["cpu_util"] for r in telemetry]
        ),
        "memory_rss_mb": mean(
            [r["memory_rss_mb"] for r in telemetry]
        ),
        "gpu_util": mean(
            [r["gpu_util"] for r in telemetry]
        ),
        "gpu_memory_util": mean(
            [r["gpu_memory_util"] for r in telemetry]
        ),

        "telemetry_samples": len(telemetry),
        "valid_training": valid_training,
        "exit_code": exit_code,
    }

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=result.keys()
        )
        writer.writeheader()
        writer.writerow(result)

    telemetry_path = output.with_name(
        output.stem + "_telemetry.csv"
    )

    with telemetry_path.open("w", newline="") as f:
        fields = [
            "elapsed_since_launch_s",
            "cpu_util",
            "memory_rss_mb",
            "gpu_util",
            "gpu_memory_util",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(telemetry)

    print("\nResults")
    print("-----------------------")
    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"Telemetry: {telemetry_path}", flush=True)

    # A successful process without complete training events
    # should still be treated as an invalid experiment.
    return result, (
        exit_code if exit_code != 0
        else 0 if valid_training
        else 1
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)

    args = parser.parse_args()

    _, exit_code = run(
        command=args.command,
        output=args.output,
        cpus=args.cpus,
        gpus=args.gpus,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
