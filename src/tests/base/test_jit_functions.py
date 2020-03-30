
import pytest
from numpy import array

from src.tests.fixtures import model
from src.jit_functions import assign_conditions, update_community_probability


def test_assign_conditions(model):
    """ Concurrent conditions are assigned when the model is initialized. Make sure the appropriate amount is assigned
    """
    ages = array([0] * 5000 + [1] * 5000 + [2] * 5000)
    concurrent_conditions = assign_conditions(ages, model.rng.rand(len(ages)))

    # ----- Those ages 1 (50-64) should be assigned concurrent conditions 23.74% of the time
    age_1 = concurrent_conditions[ages == 1]
    assert sum(age_1) / len(age_1) == pytest.approx(.2374, abs=.02)

    # ----- Those ages 2 (65+) should be assigned concurrent conditions 54.97% of the time
    age_2 = concurrent_conditions[ages == 2]
    assert sum(age_2) / len(age_2) == pytest.approx(.5497, abs=.02)


def test_update_community_probability(model):
    """ Community probabilities are updated before the model runs. Movement should be based on concurrent conditions
    """
    concurrent_conditions = array([1] * model.population.shape[0])

    # Probability of movement before update:
    age_1 = model.location.location.probabilities[model.age_groups == 1]
    age_1_before = age_1.mean()
    age_2 = model.location.location.probabilities[model.age_groups == 2]
    age_2_before = age_2.mean()

    new_probabilities = \
        update_community_probability(
            cp=model.location.location.probabilities,
            age=model.age_groups,
            cc=concurrent_conditions)

    # After Updates: Probabilities should go up
    age_1 = new_probabilities[model.age_groups == 1]
    assert age_1.mean() / age_1_before == pytest.approx(2.316, abs=.01)
    age_2 = new_probabilities[model.age_groups == 2]
    assert age_2.mean() / age_2_before == pytest.approx(1.437, abs=.01)


__all__ = ['model']
