
import csv
import itertools
import math
import statistics
from pathlib import Path

from ant_system import (
    ant_system,
    calculate_length,
    validate_stsp,
)


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

SEARCH_SEEDS = [101, 202, 303, 404, 505]

PARAMS = {
    "m": 10,
    "T": 50,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.5,
    "Q": 100.0,
    "tau0": 1.0,
}

# Dataset validasi yang sama dengan BAB 3 dan 4.4.
INSTANCES = {
    "V-03": {
        "D": [
            [0, 2, 4],
            [2, 0, 3],
            [4, 3, 0],
        ],
        "expected_optimum": 9,
        "expected_unique_tours": 1,
    },
    "V-04": {
        "D": [
            [0, 2, 4, 6],
            [2, 0, 3, 5],
            [4, 3, 0, 2],
            [6, 5, 2, 0],
        ],
        "expected_optimum": 13,
        "expected_unique_tours": 3,
    },
    "V-05": {
        "D": [
            [0, 3, 7, 8, 6],
            [3, 0, 4, 6, 5],
            [7, 4, 0, 2, 7],
            [8, 6, 2, 0, 3],
            [6, 5, 7, 3, 0],
        ],
        "expected_optimum": 18,
        "expected_unique_tours": 12,
    },
}


def exact_stsp_optimum(D):
    """
    Enumerasi seluruh siklus Hamilton unik STSP.

    Kota 0 ditetapkan sebagai awal untuk
    menghilangkan duplikasi rotasi.

    Pembalikan rute dieliminasi dengan
    mensyaratkan route[1] < route[-1].

    Digunakan hanya pada instance kecil.
    """
    n = validate_stsp(D)

    best_length = math.inf
    best_route = None
    examined = 0

    for permutation in itertools.permutations(
        range(1, n)
    ):
        route = (0,) + permutation

        # Rute terbalik mewakili siklus yang sama.
        if route[1] > route[-1]:
            continue

        length = math.fsum(
            D[route[i]][route[(i + 1) % n]]
            for i in range(n)
        )

        examined += 1

        if length < best_length:
            best_length = length
            best_route = list(route)

    return best_route, best_length, examined


def write_csv(path, rows):
    if not rows:
        raise ValueError("Data eksperimen kosong.")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULT_DIR.mkdir(exist_ok=True)

    raw_rows = []
    summary_rows = []

    print("EKSPERIMEN GAP OPTIMALITAS ANT SYSTEM")
    print(f"Parameter: {PARAMS}")
    print("-" * 72)

    for dataset, info in INSTANCES.items():
        D = info["D"]
        n = validate_stsp(D)

        exact_route, optimum, tour_count = (
            exact_stsp_optimum(D)
        )

        # Pemeriksaan independen referensi optimum.
        if not math.isclose(
            optimum,
            info["expected_optimum"],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise AssertionError(
                f"Optimum {dataset} berbeda."
            )

        if tour_count != info["expected_unique_tours"]:
            raise AssertionError(
                f"Jumlah tur unik {dataset} berbeda."
            )

        print(
            f"{dataset} | n={n} | "
            f"optimum={optimum:.0f} | "
            f"tur unik={tour_count}"
        )
        print(f"Rute optimum contoh: {exact_route}")

        lengths = []
        gaps = []
        successes = 0

        for repetition, seed in enumerate(
            SEARCH_SEEDS, start=1
        ):
            result = ant_system(
                D,
                seed=seed,
                **PARAMS,
            )

            route = result.best_route
            length = result.best_length

            # Periksa validitas siklus Hamilton.
            if sorted(route) != list(range(n)):
                raise AssertionError(
                    f"Rute tidak valid: {dataset}, {seed}"
                )

            # Hitung ulang panjang siklus.
            verified_length = calculate_length(
                route, D
            )

            if not math.isclose(
                length,
                verified_length,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise AssertionError(
                    f"Panjang salah: {dataset}, {seed}"
                )

            # Solusi valid tidak boleh mengalahkan
            # optimum eksak.
            if length < optimum - 1e-9:
                raise AssertionError(
                    f"Solusi lebih pendek dari optimum: "
                    f"{dataset}, {seed}"
                )

            gap = (
                (length - optimum) / optimum
            ) * 100.0

            # Hilangkan residu floating-point kecil.
            if abs(gap) < 1e-10:
                gap = 0.0

            optimal = math.isclose(
                length,
                optimum,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )

            if optimal:
                successes += 1

            lengths.append(length)
            gaps.append(gap)

            raw_rows.append({
                "dataset": dataset,
                "n": n,
                "repetition": repetition,
                "search_seed": seed,
                "optimum": optimum,
                "as_length": length,
                "gap_pct": gap,
                "optimal_found": optimal,
                "as_route": "-".join(map(str, route)),
            })

            print(
                f"  seed={seed} | "
                f"AS={length:.0f} | "
                f"gap={gap:.4f}% | "
                f"optimum_ditemukan={optimal}"
            )

        mean_length = statistics.mean(lengths)
        mean_gap = statistics.mean(gaps)
        std_gap = statistics.stdev(gaps)

        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "unique_tours": tour_count,
            "optimum": optimum,
            "repetitions": len(SEARCH_SEEDS),
            "as_min_length": min(lengths),
            "as_mean_length": mean_length,
            "as_max_length": max(lengths),
            "min_gap_pct": min(gaps),
            "mean_gap_pct": mean_gap,
            "std_gap_pct": std_gap,
            "max_gap_pct": max(gaps),
            "optimal_found_count": successes,
            "success_rate_pct": (
                100.0 * successes / len(SEARCH_SEEDS)
            ),
        })

        print(
            f"RINGKASAN {dataset}: "
            f"mean_gap={mean_gap:.4f}%, "
            f"SD_gap={std_gap:.4f}%, "
            f"berhasil={successes}/5"
        )
        print("-" * 72)

    raw_path = RESULT_DIR / "optimality_gap_raw.csv"
    summary_path = RESULT_DIR / "optimality_gap_summary.csv"

    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)

    print("EKSPERIMEN SELESAI")
    print(f"Jumlah eksekusi AS: {len(raw_rows)}")
    print(f"Data mentah: {raw_path}")
    print(f"Ringkasan: {summary_path}")


if __name__ == "__main__":
    main()
