import argparse
import subprocess
import uuid
from pathlib import Path


JOBS_DIR = Path("jobs")
RESULTS_DIR = Path("results")

JOBS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def make_job(
    cpus,
    gpus,
    batch_size,
    workers,
    command,
    walltime="00:15:00",
):
    experiment_id = str(uuid.uuid4())[:8]

    job_file = JOBS_DIR / f"{experiment_id}.sh"
    result_file = RESULTS_DIR / f"{experiment_id}.csv"

    #
    # NOTE:
    # You may need to modify the qsub resource syntax for SCC.
    #
    script = f"""#!/bin/bash

#$ -N exp_{experiment_id}
#$ -pe omp {cpus}
#$ -l h_rt={walltime}
#$ -l gpus={gpus}
#$ -j y

echo "Experiment: {experiment_id}"
echo "Host: $(hostname)"
echo "Start: $(date)"

python worker.py \
    --cpus {cpus} \
    --gpus {gpus} \
    --batch-size {batch_size} \
    --workers {workers} \
    --output {result_file} \
    --command "{command}"

echo "End: $(date)"
"""

    job_file.write_text(script)

    return experiment_id, job_file, result_file


def submit(job_file):
    result = subprocess.run(
        ["qsub", str(job_file)],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)

    parser.add_argument(
        "--command",
        required=True,
        help="Workload to run",
    )

    args = parser.parse_args()

    experiment_id, job_file, result_file = make_job(
        args.cpus,
        args.gpus,
        args.batch_size,
        args.workers,
        args.command,
    )

    job_id = submit(job_file)

    print(f"Experiment: {experiment_id}")
    print(f"Job:        {job_id}")
    print(f"Script:     {job_file}")
    print(f"Result:     {result_file}")