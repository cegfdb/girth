import numpy as np
from scipy.stats import norm as gaussian
from scipy.special import roots_legendre


def default_options():
    """ Dictionary of options used in Girth.

    Args:
        max_iteration: [int] maximum number of iterations
            allowed during processing. (Default = 25)
        distribution: [callable] function that returns a pdf
            evaluated at quadrature points, p = f(theta).
            (Default = scipy.stats.norm(0, 1).pdf)
        quadrature_bounds: (lower, upper) bounds to limit
            numerical integration. Default = (-5, 5)
        quadrature_n: [int] number of quadrature points to use
                        Default = 61
    """
    return {"max_iteration": 25,
            "distribution": gaussian(0, 1).pdf,
            "quadrature_bounds": (-5, 5),
            "quadrature_n": 61}


def validate_estimation_options(options_dict=None):
    """ Validates an options dictionary.

    Args:
        options_dict: Dictionary with updates to default_values

    Returns:
        options_dict: Updated dictionary

    """
    validate = {'max_iteration':
                    lambda x: isinstance(x, int) and x > 0,
                'distribution':
                    callable,
                'quadrature_bounds':
                    lambda x: isinstance(x, (tuple, list)) and (x[1] > x[0]),
                'quadrature_n':
                    lambda x: isinstance(x, int) and x > 7}
    
    # A complete options dictionary
    full_options = default_options()
    
    if options_dict:
        if not isinstance(options_dict, dict):
            raise AssertionError("Options must be a dictionary got: "
                                f"{type(options_dict)}.")

        for key, value in options_dict.items():
            if not validate[key](value):
                raise AssertionError("Unexpected key-value pair: "
                                     f"{key}: {value}. Please see "
                                     "documentation for expected inputs.")

        full_options.update(options_dict)

    return full_options


def get_true_false_counts(responses):
    """ Returns the number of true and false for each item.

    Takes in a responses array and returns counts associated with
    true / false.  True is a value in the dataset which equals '1'
    and false is a value which equals '0'.  All other values are
    ignored

    Args:
        responses: [n_items x n_participants] array of response values

    Returns:
        n_false: (1d array) "false" counts per item
        n_true: (1d array) "true" counts per item
    """
    n_false = np.count_nonzero(responses == 0, axis=1)
    n_true = np.count_nonzero(responses == 1, axis=1)

    return n_false, n_true


def mml_approx(dataset, discrimination=1, scalar=None):
    """ Difficulty parameter estimates of IRT model.
    
    Analytic estimates of the difficulty parameters 
    in an IRT model assuming a normal distribution .

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        scalar: (1d array) logarithm of "false counts" to "true counts" (log(n_no / n_yes))

    Returns:
        difficulty: (1d array) difficulty estimates
    """
    if scalar is None:
        n_no, n_yes = get_true_false_counts(dataset)
        scalar = np.log(n_no / n_yes)

    return (np.sqrt(1 + discrimination**2 / 3) *
            scalar / discrimination)


def convert_responses_to_kernel_sign(responses):
    """Converts dichotomous responses to the appropriate kernel sign.

    Takes in an array of responses coded as either [True/False] or [0/1]
    and converts it into [+1 / -1] to be used during parameter estimation.

    Values that are not 0 or 1 are converted into a zero which means these
    values do not contribute to parameter estimates.  This can be used to 
    account for missing values.

    Args:
        responses: [n_items x n_participants] array of response values

    Returns:
        the_sign: (2d array) sign values associated with input responses
    """
    # The default value is now 0
    the_sign = np.zeros_like(responses, dtype='float')

    # 1 -> -1
    mask = responses == 1
    the_sign[mask] = -1

    # 0 -> 1
    mask = responses == 0
    the_sign[mask] = 1

    return the_sign


def trim_response_set_and_counts(response_sets, counts):
    """ Trims all true or all false responses from the response set/counts.

    Args:
        response_set:  (2D array) response set by persons obtained by running
                        numpy.unique
        counts:  counts associated with response set

    Returns:
        response_set: updated response set with removal of undesired response patterns
        counts: updated counts to account for removal
    """
    # Remove response sets where output is all true/false
    mask = ~(np.nanvar(response_sets, axis=0) == 0)
    response_sets = response_sets[:, mask]
    counts = counts[mask]

    return response_sets, counts


def irt_evaluation(difficulty, discrimination, thetas):
    """ Evaluation of unidimensional IRT model.

    Evaluates an IRT model and returns the exact values.  This function
    supports only unidimemsional models

    Assumes the model
        P(theta) = 1.0 / (1 + exp(discrimination * (theta - difficulty)))

    Args:
        difficulty: (1d array) item difficulty parameters
        discrimination:  (1d array | number) item discrimination parameters
        thetas: (1d array) person abilities

    Returns:
        probabilities: (2d array) evaluation of sigmoid for given inputs
    """
    # If discrimination is a scalar, make it an array
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full_like(difficulty, discrimination,
                                      dtype='float')

    kernel = difficulty[:, None] - thetas
    kernel *= discrimination[:, None]
    return 1.0 / (1 + np.exp(kernel))


def _get_quadrature_points(n, a, b):
    """ Quadrature points needed for gauss-legendre integration.

    Utility function to get the legendre points,
    shifted from [-1, 1] to [a, b]

    Args:
        n: number of quadrature_points
        a: lower bound of integration
        b: upper bound of integration

    Returns:
        quadrature_points: (1d array) quadrature_points for 
                           numerical integration

    Notes:
        A local function of the based fixed_quad found in scipy, this is
        done for processing optimization
    """
    x, _ = roots_legendre(n)
    x = np.real(x)

    # Legendre domain is [-1, 1], convert to [a, b]
    return (b - a) * (x + 1) * 0.5 + a


def _compute_partial_integral(theta, difficulty, discrimination, the_sign):
    """
    Computes the partial integral for a set of item parameters

    Args:
        theta: (array) evaluation points
        difficulty: (array) set of difficulty parameters
        discrimination: (array | number) set of discrimination parameters
        the_sign:  (array) positive or negative sign
                            associated with response vector

    Returns:
        partial_integral: (2d array) 
            integration of items defined by "sign" parameters
            axis 0: individual persons
            axis 1: evaluation points (at theta)

    Notes:
        Implicitly multiplies the data by the gaussian distribution
    """
    # Size single discrimination into full array
    if np.atleast_1d(discrimination).size == 1:
        discrimination = np.full(the_sign.shape[0], discrimination,
                                 dtype='float')

    # This represents a 3-dimensional array
    # [Response Set, Person, Theta]
    # The integration happens over response set and the result is an
    # array of [Person, Theta]
    kernel = the_sign[:, :, None] * np.ones((1, 1, theta.size))
    kernel *= discrimination[:, None, None]
    kernel *= (theta[None, None, :] - difficulty[:, None, None])

    return (1.0 / (1.0 + np.exp(kernel))).prod(axis=0).squeeze()
