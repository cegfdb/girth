import numpy as np

from scipy.optimize import fminbound

import jax.numpy as jnp
from jax import jacfwd, jacrev

from girth import (trim_response_set_and_counts,
                   validate_estimation_options)


def _symmetric_functions(betas, _nplib=np):
    """Computes the symmetric functions based on the betas

        Indexes by score, left to right

    """
    polynomials = _nplib.column_stack((_nplib.ones_like(betas), 
                                       _nplib.exp(-betas)))

    # This is an easy way to compute all the values at once,
    # not necessarily the fastest
    otpt = np.array([1])
    for polynomial in polynomials:
        otpt = _nplib.convolve(otpt, polynomial)
    return otpt


def _hess_conditional(dataset):
    """Hessian function for computing standard errors."""
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)

    response_set_sums = unique_sets.sum(axis=0)

    def _hessian_calculation(solution):
        full_convolution = _symmetric_functions(solution, _nplib=jnp)

        denominator = full_convolution[response_set_sums]

        return (jnp.sum(unique_sets * solution[:,None], axis=0).dot(counts) + 
                jnp.log(denominator).dot(counts))

    return _hessian_calculation


def standard_errors_conditional(dataset, solution):
    """Computes standard errors of item parameters using hessian method.

    This function computes the hessian about the given solution, it is faster than
    the bootstrap method but still might be slow on a single core machine.

    Args:
        dataset: (2d array) of collected data
        solution: difficulty parameters
        options: dictionary with updates to default options

    Returns:
        solution: original solution given as an input
        hessian: (2d array) of partial derivatives
        standard_error: parameters from dataset without resampling
        confidence_interval: arrays with 95th percentile confidence intervals

    """
    # Compute the hessian
    hess_func = _hess_conditional(dataset)
    hessian = jacfwd(jacrev(hess_func))(solution)

    ses = list()
    for ndx in range(hessian.shape[0]):
        tmp = np.delete(np.delete(hessian, ndx, axis=0), ndx, axis=1)
        se = np.sqrt(np.diag(np.linalg.inv(tmp)).mean())
        ses.append(se)

    ci_low = solution - 1.96 * np.array(ses)
    ci_high = solution + 1.96 * np.array(ses)

    return {'Solution': solution,
            'Hessian': hessian,
            'Standard Error': ses,
            'Confidence Interval': list(zip(ci_low, ci_high))}


def rasch_conditional(dataset, discrimination=1, options=None):
    """ Estimates the difficulty parameters in a Rasch IRT model

    Args:
        dataset: [items x participants] matrix of True/False Values
        discrimination: scalar of discrimination used in model (default to 1)
        options: dictionary with updates to default options

    Returns:
        difficulty: (1d array) estimates of item difficulties

    Options:
        * max_iteration: int

    Notes:
        This function sets the sum of difficulty parameters to 
        zero for identification purposes
    """
    options = validate_estimation_options(options)
    n_items = dataset.shape[0]
    unique_sets, counts = np.unique(dataset, axis=1, return_counts=True)

    # Initialize all the difficulty parameters to zeros
    # Set an identifying_mean to zero
    ##TODO: Add option to specifiy position
    betas = np.zeros((n_items, ))
    identifying_mean = 0.0

    # Remove the zero and full count values
    unique_sets, counts = trim_response_set_and_counts(unique_sets, counts)

    response_set_sums = unique_sets.sum(axis=0)

    for iteration in range(options['max_iteration']):
        previous_betas = betas.copy()

        for ndx in range(n_items):
            partial_conv = _symmetric_functions(np.delete(betas, ndx))

            def min_func(estimate):
                betas[ndx] = estimate
                full_convolution = np.convolve([1, np.exp(-estimate)], partial_conv)

                denominator = full_convolution[response_set_sums]

                return (np.sum(unique_sets * betas[:,None], axis=0).dot(counts) + 
                        np.log(denominator).dot(counts))

            # Solve for the difficulty parameter
            betas[ndx] = fminbound(min_func, -5, 5)

            # recenter
            betas += (identifying_mean - betas.mean())

        # Check termination criterion
        if np.abs(betas - previous_betas).max() < 1e-3:
            break

    return betas / discrimination
