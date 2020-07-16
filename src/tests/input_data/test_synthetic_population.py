from src.tests.fixtures import model, synthetic_population_orange


def test_values(model, synthetic_population_orange):

    sp = synthetic_population_orange

    # ----- Orange population us around 135k
    assert len(sp) < 140000
    assert len(sp) > 130000

    # ----- All types should be represented
    assert set(sp["Age"]) == {0, 1, 2}
    assert set(sp["Sex"]) == {1, 2}
    assert set(sp["Race"]) == {1, 2, 3}
    assert set(sp["County_Code"]) == {135}

    # ----- Specific Columns should exist
    print(sp.columns)
    assert "logrecno" in sp.columns

    # ----- Agents should start in several places
    initials = sp[sp.Start_Location != 0].Start_Location.value_counts()
    assert len(initials) > 10

    # ----- The most common hospital should be UNC
    for item in model.location.locations.categories["UNC"]["ids"]:
        if "University" in model.location.locations.facilities[item]["name"]:
            unc_ch = model.location.locations.facilities[item]["int"]
    assert initials[unc_ch] == max(initials)


__all__ = ["model", "synthetic_population_orange"]
