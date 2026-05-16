#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


BYTES_PER_MIB = 1024 * 1024
DEFAULT_HEARTBEAT_INTERVAL = 0.5


def parse_migrater_metrics(
    file_path: str,
    fallback_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
):
    """
    Parse migrater logs and extract effective throughput metrics.

    Expected log format example:

        [migrater] timestamp=1714824000.123, migrating=True,
        configured=1000000 B/s, rate=1000029 B/s,
        pending=22495232 bytes ([57.09%] 29933568/52428800 bytes copied)

    If no timestamp is found, a synthetic timestamp is generated using
    fallback_interval.
    """

    timestamp_pattern = re.compile(r"timestamp=([\d.]+)")
    migrating_pattern = re.compile(r"migrating=(True|False)")
    rate_pattern = re.compile(r"\brate=(\d+(?:\.\d+)?) B/s")
    pending_pattern = re.compile(r"pending=(\d+) bytes")
    copied_pattern = re.compile(
        r"\[(\d+(?:\.\d+)?)%\]\s+(\d+)/(\d+) bytes copied"
    )

    timestamps = []
    effective_rates_mib = []
    pending_bytes = []
    copied_bytes = []
    total_bytes = []
    percentages = []

    start_timestamp = None
    synthetic_timestamp = 0.0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            migrating_match = migrating_pattern.search(line)

            if not migrating_match:
                continue

            timestamp_match = timestamp_pattern.search(line)
            rate_match = rate_pattern.search(line)
            pending_match = pending_pattern.search(line)
            copied_match = copied_pattern.search(line)

            if timestamp_match:
                current_timestamp = float(timestamp_match.group(1))

                if start_timestamp is None:
                    start_timestamp = current_timestamp

                relative_time = current_timestamp - start_timestamp
            else:
                relative_time = synthetic_timestamp
                synthetic_timestamp += fallback_interval

            is_migrating = migrating_match.group(1) == "True"

            if is_migrating and rate_match:
                effective_rate_mib = float(rate_match.group(1)) / BYTES_PER_MIB
            else:
                effective_rate_mib = 0.0

            if pending_match:
                pending = int(pending_match.group(1))
            else:
                pending = None

            if copied_match:
                percentage = float(copied_match.group(1))
                copied = int(copied_match.group(2))
                total = int(copied_match.group(3))
            else:
                percentage = None
                copied = None
                total = None

            timestamps.append(relative_time)
            effective_rates_mib.append(effective_rate_mib)
            pending_bytes.append(pending)
            copied_bytes.append(copied)
            total_bytes.append(total)
            percentages.append(percentage)

    return {
        "time": timestamps,
        "effective_rate_mib": effective_rates_mib,
        "pending_bytes": pending_bytes,
        "copied_bytes": copied_bytes,
        "total_bytes": total_bytes,
        "percentages": percentages,
    }


def make_step_series(
    x: list[float],
    y: list[float],
    interval: float = DEFAULT_HEARTBEAT_INTERVAL,
):
    """
    Extend the last point so the final step is visible in the plot.
    """

    if not x:
        return [], []

    x_step = list(x)
    y_step = list(y)

    x_step.append(x[-1] + interval)
    y_step.append(y[-1])

    return x_step, y_step


def plot_performance(
    metrics: dict,
    output_file: str,
    show: bool = False,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL,
):
    """
    Plot effective checkpoint migration throughput as step blocks.
    """

    x = metrics["time"]
    y = metrics["effective_rate_mib"]

    x_step, y_step = make_step_series(x, y, heartbeat_interval)

    plt.figure(figsize=(12, 6))

    plt.step(
        x_step,
        y_step,
        where="post",
        linewidth=1.5,
        label="Effective Migration Throughput (MiB/s)",
    )

    plt.fill_between(
        x_step,
        y_step,
        step="post",
        alpha=0.3,
    )

    plt.title("Checkpoint Migration Performance (Orchestrated)")
    plt.xlabel("Training Time (seconds since start)")
    plt.ylabel("Effective Throughput (MiB/s)")

    if y:
        max_y = max(y)
        if max_y > 0:
            plt.ylim(0, max_y * 1.2)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_file, dpi=150)
    print(f"Generated plot: {output_file}")

    if show:
        plt.show()


def print_summary(metrics: dict):
    """
    Print a small textual summary of the parsed migration metrics.
    """

    x = metrics["time"]
    y = metrics["effective_rate_mib"]

    if not x:
        print("No data found with the expected format.")
        return

    active_rates = [rate for rate in y if rate > 0]

    print(f"Total monitored time: {x[-1]:.2f} seconds")

    if active_rates:
        print(
            "Average effective throughput during migration: "
            f"{sum(active_rates) / len(active_rates):.2f} MiB/s"
        )
        print(
            "Maximum effective throughput observed:        "
            f"{max(active_rates):.2f} MiB/s"
        )

    copied_values = [
        value for value in metrics["copied_bytes"]
        if value is not None
    ]

    total_values = [
        value for value in metrics["total_bytes"]
        if value is not None
    ]

    if copied_values and total_values:
        print(
            "Final copied bytes:                           "
            f"{copied_values[-1]} / {total_values[-1]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate an effective throughput plot from migrater logs."
        )
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="migrater_output.log",
        help="Migrater log file to parse",
    )
    parser.add_argument(
        "--output",
        default="v2_migrater_plot.png",
        help="Output PNG file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot after saving it",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=DEFAULT_HEARTBEAT_INTERVAL,
        help=(
            "Fallback interval used when log lines do not include timestamps "
            "(default: 0.5 seconds)"
        ),
    )

    args = parser.parse_args()

    if not Path(args.log_file).is_file():
        raise FileNotFoundError(f"Log file not found: {args.log_file}")

    metrics = parse_migrater_metrics(
        args.log_file,
        fallback_interval=args.heartbeat_interval,
    )

    print_summary(metrics)

    if metrics["time"]:
        plot_performance(
            metrics,
            args.output,
            show=args.show,
            heartbeat_interval=args.heartbeat_interval,
        )
