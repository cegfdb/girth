from multiprocessing import Pool
from itertools import starmap, repeat
from functools import partial

import numpy as np

from scipy import integrate

import jax.numpy as jnp
from jax import jacfwd, jacrev

from girth import validate_estimation_options, convert_responses_to_kernel_sign
from girth.utils import _compute_partial_integral, _get_quadrature_points


def _hess_mml(dataset, options, mml_type):
    """Returns hessian function for one parameter model."""
    quad_start, quad_stop = options['quadrature_bounds']
    quad_n = options['quadrature_n']

    theta = _get_quadrature_points(quad_n, quad_start, quad_stop)
    distribution = options['distribution'](theta)
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)
    the_sign = convert_responses_to_kernel_sign(unique_sets)
    beta_ndx = -dataset.shape[0]

    if mml_type == 0:
        constant = jnp.full(dataset.shape[0], 1.0)
        def alpha_func(solution): return constant

    elif mml_type == 1:
        def alpha_func(solution): return jnp.full(
            dataset.shape[0], solution[0], dtype=jnp.float32)

    elif mml_type == 2:
        def alpha_func(solution): return solution[:dataset.shape[0]]

    else:
        raise KeyError(f"Not a valid mml_type: {mml_type}")

    def _hessian_calculation(solution):
        alpha = alpha_func(solution)
        beta = solution[beta_ndx:]

        partial_int = _compute_partial_integral(
            theta, beta, alpha, the_sign, _nplib=jnp)
        partial_int *= distribution

        otpt = integrate.fixed_quad(
            lambda x: partial_int, quad_start, quad_stop, n=quad_n)[0]
        return -jnp.log(otpt).dot(counts)

    return _hessian_calculation


def standard_errors_hessian(dataset, irt_model, solution, options=None):
    """Computes standard errors of item parameters using hessian method.

    This function computes the hessian about the given solution, it is faster than
    the bootstrap method but still might be slow on a single core machine.

    Args:
        dataset: (2d array) of collected data
        irt_model: callable irt function to apply to dataset
        solution: (tuple) parameters from dataset without resampling
        options: dictionary with updates to default options

    Returns:
        solution: original solution given as an input
        hessian: (2d array) of partial derivatives
        standard_error: parameters from dataset without resampling
        confidence_interval: arrays with 95th percentile confidence intervals

    Options:
        * quadrature_n: int
        * quadrature_bounds: (float, float)
        * distribution: callable

    Notes:
        Use partial for irt_models that take an discrimination parameter:
        irt_model = partial(rasch_mml, discrimination=1.2)
    """
    options = validate_estimation_options(options)

    # get hessian function depending on the irt model
    hess_func = {"rasch_mml": partial(_hess_mml, mml_type=0),
                 "rasch_full": partial(_hess_mml, mml_type=0),
                 "onepl_mml": partial(_hess_mml, mml_type=1),
                 'onepl_full': partial(_hess_mml, mml_type=1),
                 "twopl_mml": partial(_hess_mml, mml_type=2),
                 'twopl_full': partial(_hess_mml, mml_type=2)}[irt_model.__name__]
    hesssian_call = hess_func(dataset, options)

    # Vector of solution
    linear_solution = np.concatenate([np.ravel(data) for data in solution])

    # Compute the hessian
    hessian = jacfwd(jacrev(hesssian_call))(linear_solution)
    ses = np.sqrt(np.diag(np.linalg.inv(hessian)))

    start_ndx = 0
    ci_list = list()
    ses_list = list()
    for parameters in solution:
        stop_ndx = np.shape(np.atleast_1d(parameters))[0] + start_ndx
        ci_low = parameters - 1.96 * ses[start_ndx:stop_ndx]
        ci_high = parameters + 1.96 * ses[start_ndx:stop_ndx]

        ci_list.append(list(zip(ci_low, ci_high)))

        ses_list.append(ses[start_ndx:stop_ndx])

        # Update for next parameters
        start_ndx = stop_ndx

    return {'Solution': solution,
            'Hessian': hessian,
            'Standard Error': ses_list,
            'Confidence Interval': ci_list}


def _bootstrap_func(dataset, irt_model, options, iterations, local_seed):
    """Performs the boostrap sampling."""
    np.random.seed(local_seed)

    n_responses = dataset.shape[1]

    output_results = list()

    for _ in range(iterations[0], iterations[1]):
        # Get a sample of RESPONSES
        bootstrap_ndx = np.random.choice(
            n_responses, size=n_responses, replace=True)
        bootstrap_sample = dataset[:, bootstrap_ndx]

        # solve for parameters
        result = irt_model(bootstrap_sample, options=options)
        output_results.append(result)

    return output_results


def standard_errors_bootstrap(dataset, irt_model, solution=None, options=None, seed=None):
    """Computes standard errors of item parameters using bootstrap method.

    This function will be sloooow, it is best to use multiple processors to decrease
    the processing time.

    Args:
        dataset: Dataset to take random samples from
        irt_model: callable irt function to apply to dataset
        solution: (optional) parameters from dataset without resampling
        options: dictionary with updates to default options
        seed: (optional) random state integer to reproduce results

    Returns:
        solution: parameters from dataset without resampling
        standard_error: parameters from dataset without resampling
        confidence_interval: arrays with 95th percentile confidence intervals
        bias: mean difference of bootstrap mean and solution

    Options:
        * n_processors: int
        * bootstrap_iterations: int

    Notes:
        Use partial for irt_models that take an discrimination parameter:
        irt_model = partial(rasch_mml, discrimination=1.2)
    """
    options = validate_estimation_options(options)

    if seed is None:
        seed = np.random.randint(0, 100000, 1)[0]

    if solution is None:
        solution = irt_model(dataset, options=options)

    n_processors = options['n_processors']
    bootstrap_iterations = options['bootstrap_iterations']
    chunksize = np.linspace(0, bootstrap_iterations,
                            n_processors + 1, dtype='int')
    chunksize = list(zip(chunksize[:-1], chunksize[1:]))
    seeds = seed * np.arange(1.0, len(chunksize)+1, dtype='int')

    map_func = starmap
    if n_processors > 1:
        map_func = Pool(processes=n_processors).starmap

    # Run the bootstrap data
    results = map_func(_bootstrap_func, zip(repeat(dataset), repeat(irt_model),
                                            repeat(options), chunksize, seeds))
    results = list(results)

    # Unmap the results to compute the metrics
    ses_list = list()
    ci_list = list()
    bias_list = list()

    for p_ndx, parameter in enumerate(solution):
        temp_result = np.concatenate([list(zip(*results[ndx]))[p_ndx]
                                      for ndx in range(len(results))])

        parameter = np.atleast_1d(parameter)

        bias_list.append(np.nanmean(temp_result, axis=0) - parameter)
        ses_list.append(np.nanstd(temp_result, axis=0, ddof=0))
        if parameter.shape[0] == 1:
            ci_list.append((np.percentile(temp_result, 2.5, axis=0),
                            np.percentile(temp_result, 97.5, axis=0)))
        else:
            ci_list.append(list(zip(np.percentile(temp_result, 2.5, axis=0),
                                    np.percentile(temp_result, 97.5, axis=0))))

    return {'Solution': solution,
            'Standard Error': ses_list,
            'Confidence Interval': ci_list,
            'Bias': bias_list}


if __name__ == "__main__":
    pass
