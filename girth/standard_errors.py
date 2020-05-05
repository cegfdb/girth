from multiprocessing import Pool
from itertools import starmap, repeat

import numpy as np

from girth import validate_estimation_options


def _bootstrap_func(dataset, irt_model, options, iterations, local_seed):
    """Performs the boostrap sampling."""
    np.random.seed(local_seed)
    
    n_responses = dataset.shape[1]
    
    output_results = list()

    for _ in range(iterations[0], iterations[1]):
        # Get a sample of RESPONSES
        bootstrap_ndx = np.random.choice(n_responses, size=n_responses, replace=True)
        bootstrap_sample = dataset[:, bootstrap_ndx]

        # solve for parameters
        result = irt_model(bootstrap_sample, options=options)
        output_results.append(result)

    return output_results


def standard_errors_bootstrap(dataset, irt_model, options=None, solution=None, seed=None):
    """Computes standard errors of item parameters using bootstrap method.

    This function will be sloooow, it is best to use multiple processors to decrease
    the processing time.
    
    Args:
        dataset: Dataset to take random samples from
        irt_model: callable irt function to apply to dataset
        options: dictionary with updates to default options
        solution: (optional) parameters from dataset without resampling
        seed: (optional) random state integer to reproduce results

    Returns:
        solution: parameters from dataset without resampling
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

    # Compute the mean and confidence intervals
    samples = np.concatenate(results)
    bias = np.nanmean(samples, axis=0) - solution
    confidence_interval = list(zip(np.percentile(samples, 2.5, axis=0),
                                   np.percentile(samples, 97.5, axis=0)))

    return solution, confidence_interval, bias


if __name__ == "__main__":
    pass