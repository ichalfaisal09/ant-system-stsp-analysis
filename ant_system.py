
import math
import random
from dataclasses import dataclass


@dataclass
class ASResult:
    best_route: list[int]
    best_length: float
    pheromone: list[list[float]]


def validate_stsp(D):
    """
    Memvalidasi matriks jarak Symmetric TSP.

    Syarat:
    - Jumlah kota minimal 3.
    - Matriks persegi.
    - Diagonal bernilai nol.
    - Jarak antarkota positif dan berhingga.
    - Matriks simetris.

    Kompleksitas waktu: Theta(n^2).
    Ruang tambahan: O(1).
    """
    n = len(D)

    if n < 3:
        raise ValueError("Jumlah kota minimal 3.")

    if any(len(row) != n for row in D):
        raise ValueError("Matriks harus persegi.")

    for i in range(n):
        if not math.isfinite(D[i][i]) or D[i][i] != 0:
            raise ValueError("Diagonal harus nol.")

        for j in range(i + 1, n):
            a = D[i][j]
            b = D[j][i]

            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError("Jarak harus hingga.")

            if a <= 0 or b <= 0 or a != b:
                raise ValueError(
                    "Matriks harus simetris dan positif."
                )

    return n


def calculate_length(route, D):
    """
    Menghitung panjang siklus Hamilton.

    Rute harus merupakan permutasi seluruh indeks kota.
    Sisi kembali ke kota awal dihitung secara implisit.

    Kompleksitas waktu: Theta(n).
    Ruang tambahan: O(n).
    """
    n = len(route)

    if n < 3 or len(D) != n:
        raise ValueError("Ukuran rute tidak sesuai.")

    visited = [False] * n

    # Validasi permutasi secara linear.
    for city in route:
        if (
            not isinstance(city, int)
            or isinstance(city, bool)
            or city < 0
            or city >= n
            or visited[city]
        ):
            raise ValueError("Rute bukan permutasi valid.")

        visited[city] = True

    # Evaluasi seluruh sisi, termasuk sisi penutup.
    total = math.fsum(
        D[route[i]][route[(i + 1) % n]]
        for i in range(n)
    )

    if not math.isfinite(total) or total <= 0:
        raise ArithmeticError("Panjang rute tidak valid.")

    return total


def select_next_city(
    current,
    candidates,
    tau,
    eta,
    alpha,
    beta,
    rng,
):
    """
    Memilih kota berikutnya secara probabilistik.

    Bobot transisi:
        tau[current][j]^alpha * eta[current][j]^beta

    Menggunakan log-weight dan normalisasi
    max-subtraction untuk kestabilan numerik.

    Kompleksitas waktu: Theta(c),
    dengan c jumlah kandidat.
    """
    if not candidates:
        raise ValueError("Kandidat kosong.")

    log_weights = []

    for j in candidates:
        pheromone = tau[current][j]
        heuristic = eta[current][j]

        if pheromone <= 0 or heuristic <= 0:
            raise ArithmeticError("Bobot tidak positif.")

        value = (
            alpha * math.log(pheromone)
            + beta * math.log(heuristic)
        )

        if not math.isfinite(value):
            raise ArithmeticError("Log-bobot tidak valid.")

        log_weights.append(value)

    maximum = max(log_weights)

    weights = [
        math.exp(value - maximum)
        for value in log_weights
    ]

    total = math.fsum(weights)

    if total <= 0 or not math.isfinite(total):
        raise ArithmeticError("Normalisasi gagal.")

    u = rng.random()
    cumulative = 0.0

    for city, weight in zip(candidates, weights):
        cumulative += weight / total

        if u < cumulative:
            return city

    # Perlindungan terhadap galat pembulatan.
    return candidates[-1]


def construct_route(
    start,
    n,
    tau,
    eta,
    alpha,
    beta,
    rng,
):
    """
    Membangun satu siklus Hamilton.

    Kota awal dipilih oleh pemanggil.
    Setiap langkah hanya mempertimbangkan
    kota yang belum dikunjungi.

    Rute dikembalikan tanpa mengulang kota awal.

    Kompleksitas waktu: Theta(n^2).
    Ruang tambahan: O(n).
    """
    visited = [False] * n
    visited[start] = True

    route = [start]
    current = start

    for _ in range(n - 1):
        candidates = [
            j
            for j in range(n)
            if not visited[j]
        ]

        next_city = select_next_city(
            current,
            candidates,
            tau,
            eta,
            alpha,
            beta,
            rng,
        )

        route.append(next_city)
        visited[next_city] = True
        current = next_city

    return route


def update_pheromone(
    tau,
    routes,
    lengths,
    rho,
    Q,
):
    """
    Pembaruan feromon Ant-Cycle.

    Tahapan:
    1. Evaporasi seluruh sisi tak berarah.
    2. Deposisi Q/L oleh setiap semut.
    3. Pemeriksaan nilai feromon.

    Semua semut berkontribusi, termasuk pada
    sisi terakhir menuju kota awal.

    Matriks feromon diperbarui secara in-place.

    Kompleksitas waktu: Theta(n^2 + mn).
    Ruang tambahan: O(1), di luar data masukan.
    """
    if len(routes) != len(lengths):
        raise ValueError(
            "Jumlah rute dan panjang rute tidak sama."
        )

    n = len(tau)

    # 1. Evaporasi seluruh sisi tak berarah.
    for i in range(n):
        for j in range(i + 1, n):
            value = (1.0 - rho) * tau[i][j]

            tau[i][j] = value
            tau[j][i] = value

    # 2. Deposisi oleh semua semut (Ant-Cycle).
    for route, length in zip(routes, lengths):
        delta = Q / length

        for index in range(n):
            i = route[index]
            j = route[(index + 1) % n]

            value = tau[i][j] + delta

            tau[i][j] = value
            tau[j][i] = value

    # 3. Memastikan feromon tetap positif
    #    dan berhingga pada semua sisi.
    for i in range(n):
        for j in range(i + 1, n):
            value = tau[i][j]

            if value <= 0 or not math.isfinite(value):
                raise ArithmeticError(
                    "Feromon keluar dari domain valid."
                )

    return tau


def ant_system(
    D,
    m=10,
    T=50,
    alpha=1.0,
    beta=2.0,
    rho=0.5,
    Q=100.0,
    tau0=1.0,
    seed=101,
):
    """
    Ant System dengan mekanisme Ant-Cycle
    untuk Symmetric Traveling Salesman Problem.

    Parameter:
        D     : matriks jarak STSP
        m     : jumlah semut
        T     : jumlah iterasi
        alpha : pengaruh feromon
        beta  : pengaruh heuristik
        rho   : laju evaporasi
        Q     : konstanta deposisi
        tau0  : feromon awal
        seed  : seed generator acak

    Keluaran:
        ASResult(
            best_route,
            best_length,
            pheromone,
        )

    Kompleksitas waktu:
        Theta(T * m * n^2)

    Kompleksitas ruang total:
        Theta(n^2 + m * n)

    Tidak menyimpan seluruh histori evaluasi.
    """
    # Validasi dataset.
    n = validate_stsp(D)

    # Validasi jumlah semut dan iterasi.
    if (
        isinstance(m, bool)
        or isinstance(T, bool)
        or not isinstance(m, int)
        or not isinstance(T, int)
        or m < 1
        or T < 1
    ):
        raise ValueError(
            "m dan T harus bilangan bulat positif."
        )

    # Validasi parameter numerik.
    params = (alpha, beta, rho, Q, tau0)

    if not all(math.isfinite(x) for x in params):
        raise ValueError("Parameter harus hingga.")

    if (
        alpha < 0
        or beta < 0
        or not 0 < rho < 1
        or Q <= 0
        or tau0 <= 0
    ):
        raise ValueError("Parameter Ant System tidak sah.")

    # Generator acak lokal untuk reproduksibilitas.
    rng = random.Random(seed)

    # Inisialisasi matriks feromon.
    tau = [
        [
            0.0 if i == j else float(tau0)
            for j in range(n)
        ]
        for i in range(n)
    ]

    # Inisialisasi matriks heuristik.
    eta = [
        [
            0.0 if i == j else 1.0 / D[i][j]
            for j in range(n)
        ]
        for i in range(n)
    ]

    best_route = []
    best_length = math.inf

    # Iterasi utama Ant System.
    for _ in range(T):
        # Hanya menyimpan data iterasi saat ini.
        routes = []
        lengths = []

        # Konstruksi dan evaluasi seluruh semut.
        # Semua semut menggunakan matriks tau yang sama.
        for _ in range(m):
            start = rng.randrange(n)

            route = construct_route(
                start,
                n,
                tau,
                eta,
                alpha,
                beta,
                rng,
            )

            length = calculate_length(route, D)

            routes.append(route)
            lengths.append(length)

            # Menyimpan solusi terbaik global
            # sepanjang proses pencarian.
            if length < best_length:
                best_length = length
                best_route = route.copy()

        # Pembaruan hanya dilakukan setelah
        # seluruh semut menyelesaikan konstruksi.
        tau = update_pheromone(
            tau,
            routes,
            lengths,
            rho,
            Q,
        )

    return ASResult(
        best_route=best_route,
        best_length=best_length,
        pheromone=tau,
    )
