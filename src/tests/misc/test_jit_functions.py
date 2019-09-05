
import pytest
from numpy import array
import sys
sys.path.append("")

from src.tests.fixtures import model
from src.jit_functions import find_probability, select_agents, assign_conditions
from src.cdi_functions import update_community_probability


def test_find_demo_id(model):
    # find_demo_id
    print(model)
    #  TODO


def test_find_probability(model):

    probs = find_probability(model.community_probability, array([i for i in range(1800)]))

    assert len(probs) == 1800
    assert max(probs) < 1


def test_select_agents(model):

    use_agents = [item for item in range(model.population.shape[0])]
    values = model.data[:, model.columns['community_probability']][use_agents]
    # Most agents should move
    move_agents = select_agents(values, model.np_rng.rand(values.shape[0], 1))
    assert len(move_agents) == model.population.shape[0]
    assert sum(move_agents) < 5

    # If we up their probability, they should move
    values = array([.95] * model.population.shape[0])
    move_agents = select_agents(values, model.np_rng.rand(values.shape[0], 1))
    assert len(move_agents) == model.population.shape[0]
    assert sum(move_agents) > 75


def test_assign_conditions(model):

    ages = array([0] * 5000 + [1] * 5000 + [2] * 5000)

    concurrent_conditions = assign_conditions(ages, model.np_rng.rand(len(ages), 1))

    # ----- Those ages 1 (50-64) should be assigned concurrent conditions 23.74% of the time
    age_1 = concurrent_conditions[ages == 1]
    assert sum(age_1) / len(age_1) == pytest.approx(.2374, abs=.02)

    # ----- Those ages 2 (65+) should be assigned concurrent conditions 54.97% of the time
    age_2 = concurrent_conditions[ages == 2]
    assert sum(age_2) / len(age_2) == pytest.approx(.5497, abs=.02)


def test_update_community_probability(model):

    concurrent_conditions = array([1] * model.population.shape[0])

    # Before update:
    age_1 = model.data[:, model.columns['community_probability']][model.data[:, model.columns['age_group']] == 1]
    age_1_before = age_1.mean()
    age_2 = model.data[:, model.columns['community_probability']][model.data[:, model.columns['age_group']] == 2]
    age_2_before = age_2.mean()

    new_probabilities = \
        update_community_probability(
            cp=model.data[:, model.columns['community_probability']],
            age=model.data[:, model.columns['age_group']],
            cc=concurrent_conditions)

    # After Updates: Probabilities should go up
    age_1 = new_probabilities[model.data[:, model.columns['age_group']] == 1]
    assert age_1.mean() / age_1_before == pytest.approx(2.316, abs=.01)
    age_2 = new_probabilities[model.data[:, model.columns['age_group']] == 2]
    assert age_2.mean() / age_2_before == pytest.approx(1.437, abs=.01)


__all__ = ['model']
