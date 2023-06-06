"""
A middleware for accessing em-bright from read-rapidpe.

Several functions are copied/modified from ligo.em_bright:
    https://git.ligo.org/emfollow/em-properties/em-bright/-/blob/main/ligo/em_bright/em_bright.py
"""


import h5py
import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology, units as u

from ligo.em_bright import computeDiskMass, utils
ALL_EOS_DRAWS = utils.load_eos_posterior()


def get_redshifts(distances, N=10000):
    """
    Compute redshift using the Planck15 cosmology.

    Parameters
    ----------
    distances: float or numpy.ndarray
              distance(s) in Mpc

    N : int, optional
      Number of steps for the computation of the interpolation function

    Example
    -------
    >>> distances = np.linspace(10, 100, 10)
    >>> em_bright.get_redshifts(distances)
    array([0.00225566, 0.00450357, 0.00674384, 0.00897655,
           0.01120181, 0.0134197 , 0.01563032, 0.01783375
           0.02003009, 0.02221941])

    Notes
    -----
    This function accepts HDF5 posterior samples file and computes
    redshift by interpolating the distance-redshift relation.
    """
    function = cosmology.Planck15.luminosity_distance
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    z_min = cosmology.z_at_value(func=function, fval=min_dist*u.Mpc)
    z_max = cosmology.z_at_value(func=function, fval=max_dist*u.Mpc)
    z_steps = np.linspace(z_min - (0.1*z_min), z_max + (0.1*z_max), N)
    lum_dists = cosmology.Planck15.luminosity_distance(z_steps)
    s = interp1d(lum_dists, z_steps)
    redshifts = s(distances)
    return redshifts


def em_bright(result, threshold=3.0, num_eos_draws=1000, eos_seed=None):
    """
    Compute ``HasNS``, ``HasRemnant``, and ``HasMassGap`` probabilities
    from read-rapidpe result.

    Parameters
    ----------
    result : RapidPEresult
        RapidPE result object

    threshold : float, optional
        Maximum neutron star mass for `HasNS` computation

    num_eos_draws : int
        providing an int here runs eos marginalization
        with the value determining how many eos's to draw

    eos_seed : int
        seed for random eos draws

    Returns
    -------
    dict
        {HasNS, HasRemnant, HasMassGap} predicted values.

    """

    has_ns, has_remnant, has_massgap = \
        source_classification_samples(samples=result.posterior_samples,
                                      threshold=threshold,
                                      num_eos_draws=num_eos_draws,
                                      eos_seed=eos_seed)
    return ({
        'HasNS': has_ns,
        'HasRemnant': has_remnant,
        'HasMassGap': has_massgap
    })


def source_classification_pe(posterior_samples_file, threshold=3.0,
                             num_eos_draws=None, eos_seed=None):

    with h5py.File(posterior_samples_file, 'r') as data:
        samples = data['posterior_samples'][()]

    return source_classification_samples(samples=samples,
                                         threshold=threshold,
                                         num_eos_draws=num_eos_draws,
                                         eos_seed=eos_seed)


def source_classification_samples(samples, threshold=3.0,
                                  num_eos_draws=None, eos_seed=None):
    """
    Compute ``HasNS``, ``HasRemnant``, and ``HasMassGap`` probabilities
    from posterior samples.

    Parameters
    ----------
    samples : dict or recarray
        Posterior samples

    threshold : float, optional
        Maximum neutron star mass for `HasNS` computation

    num_eos_draws : int
        providing an int here runs eos marginalization
        with the value determining how many eos's to draw

    eos_seed : int
        seed for random eos draws

    Returns
    -------
    tuple
        (HasNS, HasRemnant, HasMassGap) predicted values.

    """

    try:
        mass_1, mass_2 = samples['mass_1_source'], samples['mass_2_source']
    except ValueError:
        lum_dist = samples['luminosity_distance']
        redshifts = get_redshifts(lum_dist)
        try:
            mass_1, mass_2 = samples['mass_1'], samples['mass_2']
            mass_1, mass_2 = mass_1/(1 + redshifts), mass_2/(1 + redshifts)
        except ValueError:
            chirp_mass, mass_ratio = samples['chirp_mass'], samples['mass_ratio']  # noqa:E501
            chirp_mass = chirp_mass/(1 + redshifts)
            mass_1 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(-3/5)  # noqa:E501
            mass_2 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(2/5)

    try:
        a_1 = samples["spin_1z"]
        a_2 = samples["spin_2z"]
    except ValueError:
        try:
            a_1 = samples['a_1'] * np.cos(samples['tilt_1'])
            a_2 = samples['a_2'] * np.cos(samples['tilt_2'])
        except ValueError:
            a_1, a_2 = np.zeros(len(mass_1)), np.zeros(len(mass_2))

    if num_eos_draws:
        np.random.seed(eos_seed)
        prediction_nss, prediction_ems = [], []

        m1, m2, a1, a2 = mass_1, mass_2, a_1, a_2
        rand_subset = np.random.choice(
            len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS))  # noqa:E501
        subset_draws = ALL_EOS_DRAWS[rand_subset]
        M, R = subset_draws['M'], subset_draws['R']
        max_masses = np.max(M, axis=1)
        f_M = [interp1d(m, r, bounds_error=False) for m, r in zip(M, R)]
        for mass_radius_relation, max_mass in zip(f_M, max_masses):
            M_rem = computeDiskMass.computeDiskMass(m1, m2, a1, a2, eosname=mass_radius_relation, max_mass=max_mass)  # noqa:E501
            prediction_nss.append(np.mean(m2 <= max_mass))
            prediction_ems.append(np.mean(M_rem > 0))

        prediction_ns = np.mean(prediction_nss)
        prediction_em = np.mean(prediction_ems)

    else:
        M_rem = computeDiskMass.computeDiskMass(mass_1, mass_2, a_1, a_2)
        prediction_ns = np.sum(mass_2 <= threshold)/len(mass_2)
        prediction_em = np.sum(M_rem > 0)/len(M_rem)

    prediction_mg = (mass_1 <= 5) & (mass_1 >= 3)
    prediction_mg += (mass_2 <= 5) & (mass_2 >= 3)
    prediction_mg = np.sum(prediction_mg)/len(mass_1)

    return prediction_ns, prediction_em, prediction_mg
