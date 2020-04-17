import pytest
from pandas import DataFrame

from src.tests.fixtures import model, transition_files, id_files


def test_community(transition_files: DataFrame):
    c = transition_files["community"]
    # ----- 100 Counties, and 3 age groups
    assert c.shape[0] == 300

    # ----- The sum of UNC, Large, Small, LT, and NH should be the probability
    assert (
        max(c.Probability - c[["UNC", "LARGE", "SMALL", "LT", "NH"]].sum(axis=1))
        < 1 * 10 ** -7
    )

    # ----- Agents cannot move to the community from the community
    value = c["COMMUNITY"].unique()
    assert len(value) == 1
    assert value[0] == 0

    # ----- Only 65+ For nursing home
    assert max(c[c.Age < 2].NH) == 0
    assert min(c[c.Age == 2].NH) > 0


def test_discharge_files(transition_files, id_files):
    hospitals = id_files["hospitals"]
    for file_name in ["LARGE", "SMALL", "UNC"]:
        # ----- There are 100 counties
        file = transition_files[file_name]
        assert len(file) == 100

        # ----- Make sure all hospitals are represented
        ids = hospitals[hospitals.Category == file_name].ID.values
        for hospital_id in ids:
            assert hospital_id in file.columns

        # ----- Rows should sum to either 0 or 1
        values = round(file[ids].sum(axis=1), 5).unique()
        assert 0 in values
        assert 1 in values


def test_nh(model, transition_files):
    file = transition_files["location"]
    nh = file[file.Facility == "NH"]

    # ----- Age should only be 2
    value = nh["Age"].unique()
    assert len(value) == 1
    assert value[0] == 2

    # ----- Rows should add to 1
    value = round(
        nh[list(model.location.locations.categories.keys())].sum(axis=1), 5
    ).unique()
    assert len(value) == 1
    assert value[0] == 1

    # ----- Community, LT, and NH all have specific values
    assert nh.COMMUNITY.mean() == pytest.approx(0.673, 0.1)
    assert nh.LT.mean() == 0
    assert nh.NH.mean() == 0


def test_lt(model, transition_files):
    file = transition_files["location"]
    lt = file[file.Facility == "LT"]

    # ----- All Ages can go to LT
    value = lt["Age"].unique()
    assert len(value) == 3

    # ----- Rows should add to 1
    value = round(
        lt[list(model.location.locations.categories.keys())].sum(axis=1), 5
    ).unique()
    assert len(value) == 1
    assert value[0] == 1

    # ----- LT can only be 0
    assert lt.LT.mean() == 0

    # ----- Only 65+ For nursing home
    assert max(lt[lt.Age < 2].NH) == 0
    assert min(lt[lt.Age == 2].NH) > 0


def test_hospitals(model, transition_files, id_files):
    hospitals = id_files["hospitals"]
    file = transition_files["location"]
    for file_name in ["LARGE", "SMALL", "UNC"]:
        hospital_ids = hospitals[hospitals.Category == file_name].ID.values
        hospital_file = file[file.Facility.isin(hospital_ids)]

        # ----- Should be all facilities for that specific category
        values = hospital_file["Facility"].unique()
        assert len(values) == len(hospital_ids)

        # ----- Rows should add to 1
        value = round(
            hospital_file[list(model.location.locations.categories.keys())].sum(axis=1),
            5,
        ).unique()
        assert len(value) == 1
        assert value[0] == 1

        # ----- Only 65+ For nursing home
        assert max(hospital_file[hospital_file.Age < 2].NH) == 0
        assert min(hospital_file[hospital_file.Age == 2].NH) > 0


__all__ = ["model", "transition_files", "id_files"]
