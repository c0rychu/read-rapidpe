"""
A middleware for accessing em-bright from read-rapidpe.

Several functions are copied/modified from ligo.em_bright:
    https://git.ligo.org/emfollow/em-properties/em-bright/-/blob/v1.1.4.post2/ligo/em_bright/em_bright.py
"""


import h5py
import numpy as np
from scipy.interpolate import interp1d

from ligo.em_bright import computeDiskMass, utils
from ligo.em_bright.em_bright import get_redshifts
ALL_EOS_DRAWS = utils.load_eos_posterior()


def em_bright(result, threshold=3.0, num_eos_draws=2000, eos_seed=None,
              eosname=None):
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
                                      eos_seed=eos_seed,
                                      eosname=eosname)
    return ({
        'HasNS': has_ns,
        'HasRemnant': has_remnant,
        'HasMassGap': has_massgap
    })


def source_classification_pe(posterior_samples_file, threshold=3.0,
                             num_eos_draws=10000, eos_seed=None,
                             eosname=None):
    """
    Compute ``HasNS``, ``HasRemnant``, and ``HasMassGap`` probabilities
    from posterior samples.

    Parameters
    ----------
    posterior_samples_file : str
        Posterior samples file

    threshold : float, optional
        Maximum neutron star mass for `HasNS` computation

    num_eos_draws : int
        providing an int here runs eos marginalization
        with the value determining how many eos's to draw

    eos_seed : int
        seed for random eos draws

    eosname : str
        Equation of state name, inferred from ``lalsimulation``. Supersedes
        eos marginalization method when provided.

    Returns
    -------
    tuple
        (HasNS, HasRemnant, HasMassGap) predicted values.

    """

    with h5py.File(posterior_samples_file, 'r') as data:
        samples = data['posterior_samples'][()]

    return source_classification_samples(samples=samples,
                                         threshold=threshold,
                                         num_eos_draws=num_eos_draws,
                                         eos_seed=eos_seed,
                                         eosname=eosname)


def source_classification_samples(samples, threshold=3.0,
                                  num_eos_draws=10000, eos_seed=None,
                                  eosname=None):
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

    eosname : str
        Equation of state name, inferred from ``lalsimulation``. Supersedes
        eos marginalization method when provided.

    Returns
    -------
    tuple
        (HasNS, HasRemnant, HasMassGap) predicted values.

    """

    try:
        mass_1, mass_2 = samples['mass_1_source'], samples['mass_2_source']
    except (ValueError, KeyError):
        lum_dist = samples['luminosity_distance']
        redshifts = get_redshifts(lum_dist)
        try:
            mass_1, mass_2 = samples['mass_1'], samples['mass_2']
            mass_1, mass_2 = mass_1/(1 + redshifts), mass_2/(1 + redshifts)
        except (ValueError, KeyError):
            chirp_mass, mass_ratio = samples['chirp_mass'], samples['mass_ratio']  # noqa:E501
            chirp_mass = chirp_mass/(1 + redshifts)
            mass_1 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(-3/5)  # noqa:E501
            mass_2 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(2/5)

    try:
        a_1 = samples["spin_1z"]
        a_2 = samples["spin_2z"]
    except (ValueError, KeyError):
        try:
            a_1 = samples['a_1'] * np.cos(samples['tilt_1'])
            a_2 = samples['a_2'] * np.cos(samples['tilt_2'])
        except (ValueError, KeyError):
            a_1, a_2 = np.zeros(len(mass_1)), np.zeros(len(mass_2))

    if eosname:
        M_rem = computeDiskMass.computeDiskMass(mass_1, mass_2, a_1, a_2,
                                                eosname=eosname)
        prediction_ns = np.sum(mass_2 <= threshold)/len(mass_2)
        prediction_em = np.sum(M_rem > 0)/len(M_rem)

    else:
        np.random.seed(eos_seed)
        prediction_nss, prediction_ems = [], []
        # EoS draws from: 10.5281/zenodo.6502467
        rand_subset = np.random.choice(
            len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS), replace=False)  # noqa:E501
        subset_draws = ALL_EOS_DRAWS[rand_subset]
        # convert radius to m from km
        M, R = subset_draws['M'], 1000*subset_draws['R']
        max_masses = np.max(M, axis=1)
        f_M = [interp1d(m, r, bounds_error=False) for m, r in zip(M, R)]
        for mass_radius_relation, max_mass in zip(f_M, max_masses):
            M_rem = computeDiskMass.computeDiskMass(mass_1, mass_2, a_1, a_2, eosname=mass_radius_relation, max_mass=max_mass)  # noqa:E501
            prediction_nss.append(np.mean(mass_2 <= max_mass))
            prediction_ems.append(np.mean(M_rem > 0))

        prediction_ns = np.mean(prediction_nss)
        prediction_em = np.mean(prediction_ems)

    prediction_mg = (mass_1 <= 5) & (mass_1 >= 3)
    prediction_mg += (mass_2 <= 5) & (mass_2 >= 3)
    prediction_mg = np.sum(prediction_mg)/len(mass_1)

    return prediction_ns, prediction_em, prediction_mg
