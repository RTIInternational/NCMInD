from src.tests.fixtures import model

# def test_risk_transitions(model):

#     rt = create_cdi_dictionary(model)

#     keys = list(rt.keys())

#     # Locations
#     locations = [item[0] for item in keys]
#     assert set(locations) == set(model.locations.values.values())

#     # Concurrent Conditions
#     cc = [item[1] for item in keys]
#     assert set(cc) == {0, 1}

#     # Antibiotic State
#     antibiotic_state = [item[2] for item in keys]
#     assert set(antibiotic_state) == {0, 1}

#     # Recent CDI Count
#     recent_cdi = [item[3] for item in keys]
#     assert set(recent_cdi) == {0, 1, 2, 3}

#     # Age
#     ages = [item[4] for item in keys]
#     assert set(ages) == {0, 1, 2}

#     # CDI State
#     cdi = [item[5] for item in keys]
#     assert set(cdi) == {0, 1, 2}  # {'CDI', 'COLONIZED', 'SUSCEPTIBLE'}

#     # Risk Ratio
#     risk_ratio = [item[6] for item in keys]
#     assert set(risk_ratio) == {1, 2, 5, 12}

#     # Test All rows add to 1
#     for key in keys:
#         assert sum(rt[key]['movement_array']) == pytest.approx(1, .001)

#     # Test the shape
#     assert len(keys) == 14 * 2 * 2 * 4 * 3 * 3 * 4


__all__ = ["model"]
