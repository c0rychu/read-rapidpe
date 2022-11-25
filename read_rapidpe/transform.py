"""
Utility functions of coordinate transformation.
Some of them are copied from RapidPE repo.
TODO: Write my own version.
"""

import numpy as np


def jacobian_m1m2_by_mceta(mc, eta):
    """
    return the Jacobian (m1, m2)/(mc, eta)
    """
    assert np.all(mc > 0), "chirp_mass (Mc) should > 0"
    assert np.all(eta > 0), "symmetric_mass_ratio (eta) should > 0"
    assert np.all(eta <= 0.25), "symmetric_mass_ratio (eta) should <= 0.25"

    return mc / (np.sqrt(1. - 4.*eta) * eta**1.2)  # 6/5 = 1.2


def jacobian_mceta_by_m1m2(m1, m2):
    """
    return the Jacobian (mc, eta)/(m1, m2)
    """
    return (m1-m2)*(m1*m2)**0.6 / (m1+m2)**3.2  # 3/5 = 0.6, 16/5 = 3.2


# =====================================================================
# This Section is copied from
# https://git.ligo.org/rapidpe-rift/rapidpe/-/blob/master/rapid_pe/lalsimutils.py
# =====================================================================

def norm_sym_ratio(eta):

    # Assume floating point precision issues
    # if np.any(np.isclose(eta, 0.25)):
    #     eta[np.isclose(eta, 0.25)] = 0.25

    # Assert phyisicality
    assert np.all(eta <= 0.25)

    return np.sqrt(1 - 4. * eta)


def lalsimutils_m1m2(Mc, eta):
    """Compute component masses from Mc, eta. Returns m1 >= m2"""
    m1 = 0.5*Mc*eta**(-3./5.)*(1. + norm_sym_ratio(eta))
    m2 = 0.5*Mc*eta**(-3./5.)*(1. - norm_sym_ratio(eta))
    return m1, m2


def lalsimutils_Mceta(m1, m2):
    """Compute chirp mass and symmetric mass ratio from component masses"""
    Mc = (m1*m2)**(3./5.)*(m1+m2)**(-1./5.)
    eta = m1*m2/(m1+m2)/(m1+m2)
    return Mc, eta


# =====================================================================


# =====================================================================
# This Section is copied from
# https://git.ligo.org/rapidpe-rift/rapidpe/-/blob/master/rapid_pe/amrlib.py
# =====================================================================

m1m2 = np.vectorize(lalsimutils_m1m2)


def transform_m1m2_to_mceta(m1, m2):
    return lalsimutils_Mceta(m1, m2)


def transform_mceta_to_m1m2(mc, eta):
    return m1m2(mc, eta)


# =====================================================================
