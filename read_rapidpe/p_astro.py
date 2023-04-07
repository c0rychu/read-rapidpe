"""
Author: Cory Chu <cory@gwlab.page>
"""

import numpy as np
from read_rapidpe.transform import transform_mceta_to_m1m2
from read_rapidpe.transform import jacobian_m1m2_by_mceta


class uniform_in_m1m2(object):
    def __init__(self, m_lower, m_upper):
        self.m_lower = m_lower
        self.m_upper = m_upper
        self._prior_np = np.vectorize(self._prior, excluded="self")

    def __call__(self, m1, m2):
        return self._prior_np(m1, m2)

    def _prior(self, m1, m2):
        if m1 < m2:
            return 0.
        elif m2 < self.m_lower or m1 > self.m_upper:
            return 0.
        else:
            # FIXME: is the "2." a correct normailzation?
            return 2./(self.m_upper - self.m_lower)**2


class uniform_in_m1m2_nsbh(object):
    def __init__(self, m_lower, m_mid, m_upper):
        self.m_lower = m_lower
        self.m_mid = m_mid
        self.m_upper = m_upper
        self._prior_np = np.vectorize(self._prior, excluded="self")

    def __call__(self, m1, m2):
        return self._prior_np(m1, m2)

    def _prior(self, m1, m2):
        if m1 < m2:
            return 0.
        elif m2 < self.m_lower or m1 > self.m_upper:
            return 0.
        elif m2 > self.m_mid or m1 < self.m_mid:
            return 0.
        else:
            return 1./((self.m_upper - self.m_mid)*(self.m_mid - self.m_lower))


def evidence_integral(res, prior):
    N = len(res.samples["mass_1"])
    return np.sum(prior(res.samples["mass_1"], res.samples["mass_2"]))/N


def _bayes_factor(result, prior):
    # Define Mesh in mc, eta space for likelihood integral
    mclist = np.linspace(result.chirp_mass.min()+0.0001,
                         result.chirp_mass.max()-0.0001,
                         500)
    etalist = np.linspace(result.symmetric_mass_ratio.min()+0.0001,
                          result.symmetric_mass_ratio.max()-0.0001,
                          500)
    delta_mc = mclist[1]-mclist[0]
    delta_eta = etalist[1]-etalist[0]
    mc, eta = np.meshgrid(mclist, etalist)

    # Interpolating likelihood
    res = result.copy()
    # TODO: Decide the interpolation method
    res.do_interpolate_marg_log_likelihood_m1m2(method="linear-scipy")
    # res.do_interpolate_marg_log_likelihood_m1m2(method="gaussian")

    mass1, mass2 = transform_mceta_to_m1m2(mc, eta)
    likelihood = np.exp(res.log_likelihood(mass1, mass2))
    jacobian = jacobian_m1m2_by_mceta(mc, eta)

    Z = delta_eta * delta_mc * np.sum(likelihood*jacobian)

    try:
        result.samples
    except AttributeError:
        result.generate_samples()

    bayes_factor = Z * evidence_integral(result, prior)
    return bayes_factor


def p_astro(result, ml=1, mm=3, mh=100):
    # Define Rate of BNS, NSBH, and BBH
    # rate[\alpha] = P(H_\alpha \ver H_1)
    # FIXME: Replaced by actual rate
    p_bns_in_H1 = (mm - ml)**2/2
    p_nsbh_in_H1 = (mm - ml)*(mh - mm)
    p_bbh_in_H1 = (mh - mm)**2/2

    rate = np.array([
        p_bns_in_H1,
        p_nsbh_in_H1,
        p_bbh_in_H1
    ])
    rate /= rate.sum()

    # Define prior \pi(\theta \vert H_\alpha)
    # FIXME: Replaced by priors from population model
    prior_bns = uniform_in_m1m2(ml, mm)
    prior_nsbh = uniform_in_m1m2_nsbh(ml, mm, mh)
    prior_bbh = uniform_in_m1m2(mm, mh)

    def prior_astro(m1, m2):
        prior = rate[0] * prior_bns(m1, m2) + \
                rate[1] * prior_nsbh(m1, m2) + \
                rate[2] * prior_nsbh(m1, m2)
        return prior

    # Define prior odd, i.e., P(H_1)/P(H_0)
    prior_odd = 1

    # Calculation
    try:
        result.samples
    except AttributeError:
        result.generate_samples()

    p_astro = np.array([
        evidence_integral(result, prior_bns),
        evidence_integral(result, prior_nsbh),
        evidence_integral(result, prior_bbh),
    ])
    p_astro *= rate
    p_astro /= p_astro.sum()

    p_terr = 1/(1 + _bayes_factor(result, prior_astro) * prior_odd)
    p_astro *= (1-p_terr)

    p_astro = {
        "BNS": p_astro[0],
        "NSBH": p_astro[1],
        "BBH": p_astro[2],
        "Terrestrial": p_terr,
    }

    return p_astro
