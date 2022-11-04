"""Testing Markov models."""

import math
import pytest
from markov import MarkovModel, estimate_parameters, likelihood, log_likelihood


def create_weather_mm()-> MarkovModel:
    SUNNY = 0
    CLOUDY = 1
    init_probs = [0.1, 0.9]  # it is almost always cloudy
    transitions_from_SUNNY = [0.3, 0.7]
    transitions_from_CLOUDY = [0.4, 0.6]
    transition_probs = [
        transitions_from_SUNNY,
        transitions_from_CLOUDY
    ]
    return MarkovModel(init_probs, transition_probs)

def test_empty() -> None:
    mm = create_weather_mm()
    assert pytest.approx(likelihood([], mm)) == 1

def test_initial_pro() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(likelihood([0], mm)) == 0.1
    assert pytest.approx(likelihood([1], mm)) == 0.9

def test_2_cases_mm() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(likelihood([0, 0], mm)) == 0.1 * 0.3
    assert pytest.approx(likelihood([1, 1], mm)) == 0.9 * 0.6
    assert pytest.approx(likelihood([0, 1], mm)) == 0.1 * 0.7
    assert pytest.approx(likelihood([1, 0], mm)) == 0.9 * 0.4

def test_3_cases_mm() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(likelihood([0, 0, 0], mm)) == 0.1 * 0.3 * 0.3
    assert pytest.approx(likelihood([1, 1, 1], mm)) == 0.9 * 0.6 * 0.6
    assert pytest.approx(likelihood([0, 1, 0], mm)) == 0.1 * 0.7 * 0.4
    assert pytest.approx(likelihood([0, 1, 1], mm)) == 0.1 * 0.7 * 0.6
    assert pytest.approx(likelihood([1, 0, 0], mm)) == 0.9 * 0.4 * 0.3
    assert pytest.approx(likelihood([1, 0, 1], mm)) == 0.9 * 0.4 * 0.7

def test_empty_log() -> None:
    mm = create_weather_mm()
    assert pytest.approx(log_likelihood([], mm)) == math.log(1)

def test_initial_pro_log() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(log_likelihood([0], mm)) == math.log(0.1)
    assert pytest.approx(log_likelihood([1], mm)) == math.log(0.9)

def test_2_cases_mm_log() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(log_likelihood([0, 0], mm)) == sum(map(math.log, [0.1, 0.3]))
    assert pytest.approx(log_likelihood([1, 1], mm)) == sum(map(math.log, [0.9, 0.6]))
    assert pytest.approx(log_likelihood([0, 1], mm)) == sum(map(math.log, [0.1, 0.7]))
    assert pytest.approx(log_likelihood([1, 0], mm)) == sum(map(math.log, [0.9, 0.4]))

def test_3_cases_mm_log() -> None:
    """Test your code."""
    mm = create_weather_mm()
    assert pytest.approx(log_likelihood([0, 0, 0], mm)) == sum(map(math.log, [0.1, 0.3, 0.3]))
    assert pytest.approx(log_likelihood([0, 1, 0], mm)) == sum(map(math.log, [0.1, 0.7, 0.4]))
    assert pytest.approx(log_likelihood([0, 1, 1], mm)) == sum(map(math.log, [0.1, 0.7, 0.6]))
    assert pytest.approx(log_likelihood([1, 1, 1], mm)) == sum(map(math.log, [0.9, 0.6, 0.6]))
    assert pytest.approx(log_likelihood([1, 0, 0], mm)) == sum(map(math.log, [0.9, 0.4, 0.3]))
    assert pytest.approx(log_likelihood([1, 0, 1], mm)) == sum(map(math.log, [0.9, 0.4, 0.7]))


def test_long_sequence() -> None:
    mm = create_weather_mm()
    assert pytest.approx(log_likelihood(50*[0], mm)) == sum(map(math.log, [0.1] + 49*[0.3]))
    assert pytest.approx(log_likelihood(50*[1], mm)) == sum(map(math.log, [0.9] + 49*[0.6]))
    assert pytest.approx(log_likelihood(25*[0, 1], mm)) == sum(map(math.log, [0.1] + 24*[0.7, 0.4] + [0.7]))

def test_trivial_case_estimate() -> None:
    mm = estimate_parameters([[0], [0]])
    assert mm.init_probs == [1]
    assert mm.trans == [[0]]

def test_simplest_case_estimate() -> None:
    mm = estimate_parameters([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1]])
    for obs, exp in zip(mm.init_probs, [0.5, 0.5]):
        assert obs == exp
    for obs, exp in zip( mm.trans[0], [0, 1]):
        assert obs == exp
    for obs, exp in zip( mm.trans[1], [0, 1]):
        assert obs == exp