
import math
from dataclasses import dataclass


@dataclass
class NNResult:
    route: list[int]
    length: float


def nearest_neighbor(D):
    """Nearest Neighbor untuk matriks STSP tervalidasi."""
    n = len(D)

    visited = [False] * n
    visited[0] = True

    route = [0]
    current = 0

    for _ in range(n - 1):
        next_city = None
        best_distance = math.inf

        # Urutan indeks meningkat; '<' menjaga indeks
        # terkecil ketika terdapat jarak yang sama.
        for j in range(n):
            if not visited[j] and D[current][j] < best_distance:
                best_distance = D[current][j]
                next_city = j

        if next_city is None:
            raise RuntimeError("Tidak ditemukan kandidat kota.")

        route.append(next_city)
        visited[next_city] = True
        current = next_city

    # Evaluasi siklus Hamilton, termasuk sisi penutup.
    total = 0.0

    for i in range(n - 1):
        total += D[route[i]][route[i + 1]]

    total += D[route[-1]][route[0]]

    if total <= 0 or not math.isfinite(total):
        raise ArithmeticError("Panjang rute tidak valid.")

    return NNResult(route=route, length=total)
