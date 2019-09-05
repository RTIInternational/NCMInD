
import sys
sys.path.append("")
from src.tests.fixtures import synthetic_population_orange


def test_values(synthetic_population_orange):

    sp = synthetic_population_orange

    assert sp.shape == (135463, 8)

    assert set(sp['Age']) == {0, 1, 2}

    assert set(sp['Sex']) == {1, 2}

    assert set(sp['Race']) == {1, 2, 3}

    assert set(sp['County_Code']) == {135}


def test_columns(synthetic_population_orange):
    sp = synthetic_population_orange

    assert 'latitude' in sp.columns

    assert 'longitude' in sp.columns

    assert 'Age_Years' in sp.columns


__all__ = ['synthetic_population_orange']
