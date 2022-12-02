"""
Author: Cory Chu <cory@gwlab.page>
"""

import numpy as np


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


def p_astro(result, ml=1, mm=3, mh=100):
    p_bns_in_H1 = (mm - ml)**2/2
    p_nsbh_in_H1 = (mm - ml)*(mh - mm)
    p_bbh_in_H1 = (mh - mm)**2/2

    rate = np.array([
        p_bns_in_H1,
        p_nsbh_in_H1,
        p_bbh_in_H1
    ])
    rate /= rate.sum()

    try:
        result.samples
    except AttributeError:
        result.generate_samples()

    p_astro = np.array([
        evidence_integral(result, uniform_in_m1m2(ml, mm)),
        evidence_integral(result, uniform_in_m1m2_nsbh(ml, mm, mh)),
        evidence_integral(result, uniform_in_m1m2(mm, mh)),
    ])
    p_astro *= rate
    p_astro /= p_astro.sum()

    p_astro = {
        "BNS": p_astro[0],
        "NSBH": p_astro[1],
        "BBH": p_astro[2],
    }

    return p_astro
