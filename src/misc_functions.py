from bisect import bisect


def create_cdf(p):
    """ Convert a list of probabilities, into a cumulative distribution function

    Parameters
    ----------
    p : list
        a list of probabilities that add to 1

    Returns
    -------
    cdf : the cumulative distribution function (as a list)

    """
    cdf = list()
    cdf.append(p[0])
    for i in range(1, len(p)):
        cdf.append(cdf[-1] + p[i])
    cdf[len(cdf) - 1] = 1
    return cdf


def normalize_and_create_cdf(p):
    """ Normalize a list of probabilities and create the cdf for them

    Parameters
    ----------
    p : list
        list of probabilities. May not add to 1, as one probability may have been removed. Thus, we normalize

    Return
    ------
    A CDF for the normalized probabilities

    """
    total = sum(p)
    p = [item / total for item in p]
    return create_cdf(p)


def random_selection(random, cdf, options):
    """ Given cumulative distribution function and a list of options, make a random selection

    Parameters
    ----------
    random: np.random.random()
        a random number between 0 and 1

    cdf : list
        a list containing the cumulative distribution values. Ex [0, .2, .3, .7, 1.0]

    options : list
        a list containing the options that can be selected

    Return
    ------
    a single value selection from options
    """
    return options[bisect(cdf, random)]
