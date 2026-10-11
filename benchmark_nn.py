
import csv
import statistics
import time
from pathlib import Path

from experiment_comparison import read_dataset
from nearest_neighbor import nearest_neighbor


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
SIZES = [10, 20, 40, 60, 80]
BATCHES = 5
TARGET_SECONDS = 0.25


def calibrated_repetitions(D):
    count = 1

    while True:
        start = time.perf_counter()

        for _ in range(count):
            nearest_neighbor(D)

        elapsed = time.perf_counter() - start

        if elapsed >= TARGET_SECONDS:
            return count

        count *= 2


def main():
    results = []

    print("BENCHMARK BATCH NEAREST NEIGHBOR")
    print("-" * 60)

    for n in SIZES:
        dataset = f"K-{n:02d}"
        D = read_dataset(n)

        # Pemanasan.
        for _ in range(100):
            nearest_neighbor(D)

        count = calibrated_repetitions(D)
        batch_times = []

        for batch in range(1, BATCHES + 1):
            start = time.perf_counter()

            for _ in range(count):
                result = nearest_neighbor(D)

            elapsed = time.perf_counter() - start

            # Waktu rata-rata satu pemanggilan.
            time_per_call = elapsed / count
            batch_times.append(time_per_call)

            print(
                f"{dataset} | batch={batch} | "
                f"repetitions={count} | "
                f"NN={result.length:.0f} | "
                f"waktu/panggilan={time_per_call:.9f}s"
            )

        mean = statistics.mean(batch_times)
        sd = statistics.stdev(batch_times)

        results.append({
            "dataset": dataset,
            "n": n,
            "batch_count": BATCHES,
            "calls_per_batch": count,
            "mean_seconds": mean,
            "std_seconds": sd,
        })

        print(
            f"RINGKASAN {dataset}: "
            f"mean={mean:.9f}s, "
            f"SD={sd:.9f}s"
        )
        print("-" * 60)

    path = RESULTS / "nn_batch_summary.csv"

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("BENCHMARK SELESAI")
    print("Hasil:", path)


if __name__ == "__main__":
    main()
