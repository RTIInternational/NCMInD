
import pytest
import sys
sys.path.append("")

from src.misc_functions import create_cdf


def test_create_cdf():

    p = [0, .2, .7, .1]
    cdf = create_cdf(p)

    # ----- The first value of CDF should equal the first value of p
    assert p[0] == cdf[0]

    # ----- The final value of the distribution should be 1
    assert cdf[-1] == 1

    # ----- Length of cdf should equal length of p
    assert len(cdf) == len(p)
