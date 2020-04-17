import numpy as np
from copy import deepcopy

from src.tests.fixtures import cdi_model


def test_initialization(cdi_model):
    # Randomly assign locations and then initialize antibiotics
    id_list = list(cdi_model.location.locations.facilities.keys())
    new_locations = np.random.choice(
        id_list, replace=True, size=cdi_model.population.shape[0]
    )
    new_locations = [
        cdi_model.location.locations.facilities[item]["int"] for item in new_locations
    ]
    cdi_model.location.location.values = np.array(new_locations)

    cdi_model.disease.antibiotics.initialize_probabilities()

    # Every place has antibiotics - min should be greater than 0
    assert cdi_model.disease.antibiotics.probabilities.min() > 0
    # Probabilities cannot be greater than 1
    assert cdi_model.disease.antibiotics.probabilities.max() <= 1

    # No one starts with antibiotics
    assert all(cdi_model.disease.antibiotics.values == 0)


def test_step(cdi_model):
    """ If no one has antibiotics, and the probability is high enough, people should receive antibiotics
    """
    m = deepcopy(cdi_model)
    m.create_sql_connection()
    # Scenario 1: Update antibiotic probabilities for any agent who has moved locations: TODO

    # Scenario 2: Simulate antibiotic prescription for living agents NOT ON antibiotics
    m.disease.antibiotics.values.fill(0)
    m.disease.antibiotics.probabilities = np.array([1] * cdi_model.population.shape[0])
    m.disease.antibiotics.step()
    # --- Everyone should have antibiotics
    assert m.disease.antibiotics.values.min() == 1
    # --- Should be random "antibiotics ends" days
    assert m.disease.antibiotics.ends.min() > m.time + 90
    assert m.disease.antibiotics.ends.max() < m.time + 110
    assert len(np.unique(m.disease.antibiotics.ends)) > 5

    # Scenario 3: Living agents, ON antibiotics, completed the initial course: TODO


def test_give_antibiotics(cdi_model):
    pass


def test_assign_antibiotics_type(cdi_model):
    pass


def test_create_antibiotics_type_dictionary(cdi_model):
    pass


__all__ = ["cdi_model"]
