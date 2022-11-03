"""Testing Markov models."""

import pytest
from markov import MarkovModel, likelihood


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