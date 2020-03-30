
from src.misc_functions import create_cdf, normalize_and_create_cdf, random_selection


def test_create_cdf():
    p = [0, .2, .7, .1]
    cdf = create_cdf(p)
    # ----- The first value of CDF should equal the first value of p
    assert p[0] == cdf[0]
    # ----- The final value of the distribution should be 1
    assert cdf[-1] == 1
    # ----- Length of cdf should equal length of p
    assert len(cdf) == len(p)


def test_normalize_and_create_cdf():
    p = [0, .2, .7]
    cdf = normalize_and_create_cdf(p)
    # ----- The first value of CDF should equal the first value of p
    assert p[0] == cdf[0]
    # ----- The final value of the distribution should be 1
    assert cdf[-1] == 1
    # ----- Length of cdf should equal length of p
    assert len(cdf) == len(p)


def test_random_selection():
    p = [.1, .1, .7, .1]
    options = [0, 1, 2, 3]
    cdf = create_cdf(p)
    rs = random_selection(.95, cdf, options)
    # ----- .95 is between .9 and 1.0, so 2 is selected
    assert rs == 3
    rs = random_selection(.5, cdf, options)
    # ----- .5 is between .2 and .7, so 2 is selected
    assert rs == 2
    rs = random_selection(0.11, cdf, options)
    # ----- .1 is between 0.1 and .2, so 1 is selected
    assert rs == 1
    rs = random_selection(0.09, cdf, options)
    # ----- .09 is between 0 and .1, so 0 is selected
    assert rs == 0
