
import math
import pytest
from itertools import permutations
from ant_system import (
    ant_system,
    calculate_length,
    update_pheromone,
    validate_stsp,
)


from ant_system import construct_route

class FixedRNG:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)


def test_manual_v04_route_construction():
    n = 4

    tau = [
        [0.0 if i == j else 1.0 for j in range(n)]
        for i in range(n)
    ]

    eta = [
        [0.0 if i == j else 1.0 / D4[i][j] for j in range(n)]
        for i in range(n)
    ]

    # Semut 1: A -> B -> C -> D
    rng1 = FixedRNG([0.20, 0.30, 0.50])

    route1 = construct_route(
        0, n, tau, eta, 1.0, 2.0, rng1
    )

    # Semut 2: A -> D -> B -> C
    rng2 = FixedRNG([0.95, 0.10, 0.50])

    route2 = construct_route(
        0, n, tau, eta, 1.0, 2.0, rng2
    )

    assert route1 == [0, 1, 2, 3]
    assert route2 == [0, 3, 1, 2]

    assert calculate_length(route1, D4) == 13
    assert calculate_length(route2, D4) == 18

    updated = update_pheromone(
        [row.copy() for row in tau],
        [route1, route2],
        [13, 18],
        rho=0.5,
        Q=12,
    )

    assert updated[0][1] == pytest.approx(37 / 26)
    assert updated[0][2] == pytest.approx(7 / 6)
    assert updated[0][3] == pytest.approx(163 / 78)
    assert updated[1][2] == pytest.approx(163 / 78)
    assert updated[1][3] == pytest.approx(7 / 6)
    assert updated[2][3] == pytest.approx(37 / 26)



D3 = [
    [0, 2, 4],
    [2, 0, 3],
    [4, 3, 0],
]

D4 = [
    [0, 2, 4, 6],
    [2, 0, 3, 5],
    [4, 3, 0, 2],
    [6, 5, 2, 0],
]

D5 = [
    [0, 3, 7, 8, 6],
    [3, 0, 4, 6, 5],
    [7, 4, 0, 2, 7],
    [8, 6, 2, 0, 3],
    [6, 5, 7, 3, 0],
]

def brute_force_stsp(D):
    n = len(D)
    best_length = float("inf")
    best_route = None
    checked = 0

    for perm in permutations(range(1, n)):
        # Hilangkan rute balik yang ekuivalen.
        if perm[0] > perm[-1]:
            continue

        route = [0, *perm]
        length = sum(
            D[route[i]][route[(i + 1) % n]]
            for i in range(n)
        )

        checked += 1

        if length < best_length:
            best_length = length
            best_route = route

    return best_route, best_length, checked


@pytest.mark.parametrize(
    "D,expected_length,expected_count",
    [
        (D3, 9, 1),
        (D4, 13, 3),
        (D5, 18, 12),
    ],
)
def test_brute_force_reference(
    D, expected_length, expected_count
):
    route, length, checked = brute_force_stsp(D)

    assert length == expected_length
    assert checked == expected_count
    assert_valid_route(route, len(D))

def assert_valid_route(route, n):
    assert len(route) == n
    assert sorted(route) == list(range(n))


@pytest.mark.parametrize(
    "D,n",
    [(D3, 3), (D4, 4), (D5, 5)],
)
def test_valid_stsp(D, n):
    assert validate_stsp(D) == n


def test_reject_asymmetric_matrix():
    invalid = [
        [0, 2, 4],
        [3, 0, 5],
        [4, 5, 0],
    ]

    with pytest.raises(ValueError):
        validate_stsp(invalid)


def test_reject_nonpositive_distance():
    invalid = [
        [0, 0, 4],
        [0, 0, 3],
        [4, 3, 0],
    ]

    with pytest.raises(ValueError):
        validate_stsp(invalid)


@pytest.mark.parametrize(
    "D,route,expected",
    [
        (D3, [0, 1, 2], 9),
        (D4, [0, 1, 2, 3], 13),
        (D4, [0, 3, 1, 2], 18),
        (D5, [0, 1, 2, 3, 4], 18),
    ],
)
def test_calculate_length(D, route, expected):
    assert calculate_length(route, D) == expected


@pytest.mark.parametrize(
    "D",
    [D3, D4, D5],
)
def test_ant_system_produces_valid_route(D):
    n = len(D)

    result = ant_system(
        D,
        m=10,
        T=5,
        alpha=1,
        beta=2,
        rho=0.5,
        Q=100,
        tau0=1,
        seed=101,
    )

    assert_valid_route(result.best_route, n)

    assert result.best_length == pytest.approx(
        calculate_length(result.best_route, D)
    )

    for i in range(n):
        assert result.pheromone[i][i] == 0

        for j in range(i + 1, n):
            assert result.pheromone[i][j] > 0
            assert math.isfinite(result.pheromone[i][j])
            assert result.pheromone[i][j] == pytest.approx(
                result.pheromone[j][i]
            )


def test_seed_reproducibility():
    first = ant_system(D5, m=5, T=10, seed=101)
    second = ant_system(D5, m=5, T=10, seed=101)

    assert first.best_route == second.best_route
    assert first.best_length == second.best_length
    assert first.pheromone == second.pheromone


def test_manual_ant_cycle_update():
    tau = [
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ]

    routes = [
        [0, 1, 2, 3],
        [0, 3, 1, 2],
    ]

    lengths = [13, 18]

    result = update_pheromone(
        tau, routes, lengths, rho=0.5, Q=12
    )

    expected = {
        (0, 1): 37 / 26,
        (0, 2): 7 / 6,
        (0, 3): 163 / 78,
        (1, 2): 163 / 78,
        (1, 3): 7 / 6,
        (2, 3): 37 / 26,
    }

    for (i, j), value in expected.items():
        assert result[i][j] == pytest.approx(value)
        assert result[j][i] == pytest.approx(value)


def test_reject_invalid_parameters():
    with pytest.raises(ValueError):
        ant_system(D4, m=0)

    with pytest.raises(ValueError):
        ant_system(D4, T=0)

    with pytest.raises(ValueError):
        ant_system(D4, rho=1.0)


def test_reject_mismatched_update_data():
    tau = [
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ]

    with pytest.raises(ValueError):
        update_pheromone(
            tau,
            routes=[[0, 1, 2]],
            lengths=[],
            rho=0.5,
            Q=12,
        )
