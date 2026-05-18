#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"timestamp=(?P<timestamp>\d+(?:\.\d+)?),\s*"
    r"migrating=(?P<migrating>True|False),\s*"
    r"epoch=(?P<epoch>\d+),\s*"
    r"total_epochs=(?P<total_epochs>\d+),\s*"
    r"configured=(?P<configured>\d+)\s+B/s,\s*"
    r"rate=(?P<rate>\d+)\s+B/s,\s*"
    r"pending=(?P<pending>\d+)\s+bytes"
)


def parse_bandwidth(value: str) -> float:
    """
    Parse a bandwidth string into bytes per second.

    Examples:
        1000000 -> 1000000 B/s
        512MiB -> 536870912 B/s
        1GiB -> 1073741824 B/s
        500MB -> 500000000 B/s
    """

    value = value.strip()

    units = {
        "B": 1,
        "KB": 1e3,
        "MB": 1e6,
        "GB": 1e9,
        "KiB": 1024,
        "MiB": 1024 ** 2,
        "GiB": 1024 ** 3,
    }

    match = re.fullmatch(r"(?P<number>\d+(?:\.\d+)?)(?P<unit>[A-Za-z]+)?", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid bandwidth value: {value}"
        )

    number = float(match.group("number"))
    unit = match.group("unit") or "B"

    if unit not in units:
        raise argparse.ArgumentTypeError(
            f"Invalid bandwidth unit '{unit}'. Use one of: {', '.join(units)}"
        )

    return number * units[unit]


def parse_log_file(log_path: Path) -> tuple[list[float], list[float]]:
    """
    Parse a migrater log file.

    Returns:
        tuple[list[float], list[float]]:
            Relative timestamps and bandwidth rates in bytes per second.
    """

    timestamps = []
    rates = []

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = LOG_PATTERN.search(line)
            if not match:
                continue

            migrating = match.group("migrating") == "True"
            timestamps.append(float(match.group("timestamp")))
            # When the worker is idle (migrating=False), force rate to 0 so the
            # carry-forward in align_series does not keep showing the worker's
            # last observed throughput after its transfer has ended.
            rates.append(float(match.group("rate")) if migrating else 0.0)

    if not timestamps:
        raise ValueError(f"No valid migrater samples found in {log_path}")

    return timestamps, rates


def align_series(
    all_timestamps: list[list[float]],
    all_rates: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """
    Align all worker bandwidth series onto a shared relative timeline.

    This uses a step-wise carry-forward value for each worker. At each observed
    timestamp, the latest known bandwidth value for every worker is used.
    """

    start_time = min(min(timestamps) for timestamps in all_timestamps)

    events = []
    for worker_idx, (timestamps, rates) in enumerate(
        zip(all_timestamps, all_rates)
    ):
        for timestamp, rate in zip(timestamps, rates):
            events.append((timestamp, worker_idx, rate))

    events.sort(key=lambda item: item[0])

    current_rates = [0.0 for _ in all_rates]
    timeline = []
    aligned_rates = [[] for _ in all_rates]

    for timestamp, worker_idx, rate in events:
        current_rates[worker_idx] = rate

        timeline.append(timestamp - start_time)

        for idx, current_rate in enumerate(current_rates):
            aligned_rates[idx].append(current_rate)

    return timeline, aligned_rates


def to_mib_per_second(values: list[float]) -> list[float]:
    return [value / (1024 ** 2) for value in values]


def build_accumulated_rates(
    aligned_rates: list[list[float]],
) -> list[list[float]]:
    """
    Build accumulated worker bandwidth lines.

    For N workers:
        line 1 = worker 1
        line 2 = worker 1 + worker 2
        line 3 = worker 1 + worker 2 + worker 3
    """

    accumulated = []

    for idx in range(len(aligned_rates)):
        acc = [
            sum(worker_rates[sample_idx] for worker_rates in aligned_rates[:idx + 1])
            for sample_idx in range(len(aligned_rates[idx]))
        ]
        accumulated.append(acc)

    return accumulated


def plot_accumulated_bandwidth(
    log_paths: list[Path],
    orch_bw: float,
    output_path: Path,
) -> None:
    all_timestamps = []
    all_rates = []

    for log_path in log_paths:
        timestamps, rates = parse_log_file(log_path)
        all_timestamps.append(timestamps)
        all_rates.append(rates)

    timeline, aligned_rates = align_series(all_timestamps, all_rates)
    accumulated_rates = build_accumulated_rates(aligned_rates)

    plt.figure(figsize=(12, 7))

    for idx, worker_rates in enumerate(accumulated_rates):
        plt.plot(
            timeline,
            to_mib_per_second(worker_rates),
            label=f"Workers 1-{idx + 1}",
        )

    plt.axhline(
        y=orch_bw / (1024 ** 2),
        linestyle="--",
        label="Orchestrator bandwidth",
    )

    plt.xlabel("Time since first sample (s)")
    plt.ylabel("Accumulated bandwidth (MiB/s)")
    plt.title("Accumulated Migrater Bandwidth Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate accumulated bandwidth plots from migrater logs."
    )

    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="Migrater log files, one per worker, in accumulation order.",
    )

    parser.add_argument(
        "--orch-bw",
        required=True,
        type=parse_bandwidth,
        help=(
            "Orchestrator bandwidth limit. Examples: 536870912, 512MiB, "
            "1GiB, 500MB."
        ),
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("accumulated_bandwidth.png"),
        help="Output plot path. Default: accumulated_bandwidth.png",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    plot_accumulated_bandwidth(
        log_paths=args.logs,
        orch_bw=args.orch_bw,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
