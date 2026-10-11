
import pytest

from nearest_neighbor import nearest_neighbor
from ant_system import calculate_length


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


@pytest.mark.parametrize(
    "D,expected_length",
    [(D3, 9), (D4, 13), (D5, 18)],
)
def test_small_instances(D, expected_length):
    result = nearest_neighbor(D)

    assert result.route[0] == 0
    assert sorted(result.route) == list(range(len(D)))
    assert result.length == expected_length
    assert result.length == calculate_length(result.route, D)


def test_deterministic_result():
    first = nearest_neighbor(D5)
    second = nearest_neighbor(D5)

    assert first == second


def test_tie_break_smallest_index():
    D = [
        [0, 2, 2, 9],
        [2, 0, 3, 4],
        [2, 3, 0, 5],
        [9, 4, 5, 0],
    ]

    result = nearest_neighbor(D)

    # Kota 1 dan 2 sama-sama berjarak 2 dari 0.
    assert result.route[1] == 1


@pytest.mark.parametrize("D", [D3, D4, D5])
def test_valid_hamiltonian_cycle(D):
    result = nearest_neighbor(D)
    n = len(D)

    assert len(result.route) == n
    assert len(set(result.route)) == n
    assert set(result.route) == set(range(n))
    assert result.length > 0
