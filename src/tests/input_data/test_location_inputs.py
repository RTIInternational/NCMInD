
import pytest
import sys
sys.path.append("")
from src.tests.fixtures import transition_files, name_lists, initial_files


def test_files(transition_files, name_lists):

    count = 0
    for file in ['unc', 'non_unc', 'lt', 'nh', 'community']:
        file = transition_files[file]

        # Column Names
        assert file.columns.values.tolist() == name_lists['names']
        # To Values
        assert set(file['To'].values) == {'Home', 'LT', 'NH', 'Non-UNC', 'UNC'}
        # Sex
        assert set(file['Sex'].values) == {'F', 'M'}
        # Race
        assert set(file['Race'].values) == {'Black', 'Other', 'White'}
        # Age Group
        assert set(file['Age Group'].values) == {'5064', 'G65', 'L50'}
        # County Code
        assert set(file['County Code'].values) == set(range(1, 201, 2))
        # Transition Probabilities
        assert file['Transition Probability'].max() <= 1
        assert file['Transition Probability'].min() >= 0
        # Size
        assert file.shape == (9000, 18)

        # Only populations that can go to UNC should have UNC hospital data.
        c_to_unc = file[file['To'] == 'UNC'].copy()
        c_to_unc['UNC_HOSPITALS'] = c_to_unc[name_lists['hospitals']].sum(axis=1)
        if count != 0:
            # ----- Only rows were Transition Probability is greater than 0.
            can_go_to_unc = c_to_unc[c_to_unc['Transition Probability'] > 0]
            # --- All sums should be equal to 1
            assert all(can_go_to_unc['UNC_HOSPITALS'] > .9999)
            assert all(can_go_to_unc['UNC_HOSPITALS'] < 1.0001)


def test_community(transition_files):
    c = transition_files['community']
    value = c['From'].unique()
    assert len(value) == 1
    assert value[0] == 'Home'
    # --- When filtering to every demographic, the transition probability should add to 1
    # self.assertAlmostEqual(c['Transition Probability'].sum(), 1800, 4)
    # --- No one is allowed to go from Community to LT
    assert c[c['To'] == 'LT']['Transition Probability'].sum() == 0
    # --- No one is allowed to go to a NH if they are under 65
    assert c[(c['To'] == 'NH') & (c['Age Group'] != 'G65')]['Transition Probability'].sum() == 0


def test_nh(transition_files):
    nh = transition_files['nh']
    # --- From column should only equal 'NH'
    value = nh['From'].unique()
    assert len(value) == 1
    assert value[0] == 'NH'
    # --- No agents L50 or 5064 should have any probabilities
    assert nh[nh['Age Group'] != 'G65']['Transition Probability'].sum() == 0
    # --- Transition Column should add to 600
    assert nh['Transition Probability'].sum() == pytest.approx(600, .1)


def test_lt(transition_files):
    lt = transition_files['lt']
    # --- From column should only equal 'LT'
    value = lt['From'].unique()
    assert len(value) == 1
    assert value[0] == 'LT'
    # --- All Transition Columns should add to 1. No one can get stuck.
    assert lt['Transition Probability'].sum() == pytest.approx(1800, .1)


def test_unc(transition_files):
    unc = transition_files['unc']
    # --- From column should only equal 'UNC'
    value = unc['From'].unique()
    assert len(value) == 1
    assert value[0] == 'UNC'
    # --- The UNC-CH Hospital Rows should be 0. As we will use the hospital to hospital file
    # self.assertEqual(unc[self.hospitals].sum().sum(), 0)
    # --- UNC to UNC probabilities should not sum to 0
    assert unc[unc['To'] == 'UNC']['Transition Probability'].sum() > 0
    # --- All Transition Columns should add to 1. No one can get stuck.
    assert unc['Transition Probability'].sum() == pytest.approx(738, .1)


def test_non_unc(transition_files):
    non_unc = transition_files['non_unc']
    # ----- From column should only equal 'UNC'
    value = non_unc['From'].unique()
    assert len(value) == 1
    assert value[0] == 'Non-UNC'
    # --- Non-UNC to Non-UNC probabilities should not sum to 0
    assert non_unc[non_unc['To'] == 'Non-UNC']['Transition Probability'].sum() > 0
    # --- All Transition Columns should add to 1. No one can get stuck.
    assert non_unc['Transition Probability'].sum() == pytest.approx(1800, .1)


def test_unc_to_unc(transition_files, name_lists):
    unc_to_unc = transition_files['unc_to_unc']
    # Column Names:
    assert transition_files['unc_to_unc'].columns.values.tolist() == name_lists['unc_names']


def test_unc_initial(initial_files):
    unc = initial_files['unc']
    assert unc.shape == (1800, 15)
    # Sex
    assert set(unc['Sex'].values) == {'F', 'M'}
    # Race
    assert set(unc['Race'].values) == {'Black', 'Other', 'White'}
    # Age Group
    assert set(unc['Age Group'].values) == {'5064', 'L50', 'G65'}


def test_non_unc_initial(initial_files):
    non_unc = initial_files['non_unc']
    assert non_unc.shape == (1386, 5)
    # Sex
    assert set(non_unc['Sex'].values) == {'F', 'M'}
    # Race
    assert set(non_unc['Race'].values) == {'Black', 'Other', 'White'}
    # Age Group
    assert set(non_unc['Age Group'].values) == {'5064', 'L50', 'G65'}


def test_nh_initial(initial_files):
    nh = initial_files['nh']
    assert nh.shape == (1764, 6)
    # Sex
    assert set(nh['Sex'].values) == {'F', 'M'}
    # Race
    assert set(nh['Race'].values) == {'Black', 'Other', 'White'}
    # Age Group
    assert set(nh['Age Group'].values) == {'5064', 'L50', 'G65'}


def test_lt_initial(initial_files):
    lt = initial_files['lt']
    assert lt.shape == (1800, 6)
    # Sex
    assert set(lt['Sex'].values) == {'F', 'M'}
    # Race
    assert set(lt['Race'].values) == {'Black', 'Other', 'White'}
    # Age Group
    assert set(lt['Age Group'].values) == {'5064', 'L50', 'G65'}


__all__ = ['transition_files', 'name_lists', 'initial_files']
