"""
Utility functions of coordinate transformation.
"""

import numpy as np


class Mass_Spin:
    def __init__(self, obj=None):
        if obj is None:
            self._mass_1 = None
            self._mass_2 = None
            self._spin_1z = None
            self._spin_2z = None
            self.grid_coordinates = None
        else:
            self._mass_1 = obj._mass_1
            self._mass_2 = obj._mass_2
            self._spin_1z = obj._spin_1z
            self._spin_2z = obj._spin_2z
            self.grid_coordinates = obj.grid_coordinates

    def __getitem__(self, key):
        """
        Get obj.attr by obj["attr"]
        """
        return getattr(self, key)

    @classmethod
    def from_x1x2(cls, x1, x2, grid_coordinates):
        """
        Create a Mass_Spin object from the first two mass-spin coordinates
        x1, x2 defined in the "grid_coordinates"

        Parameters
        ----------
        x1: float or np.ndarray
            The first mass-spin coordinate
        x2: float or np.ndarray
            The second mass-spin coordinate
        grid_coordinates: list
            The two mass-spin coordinates defining the grid
            e.g.
                ["chirp_mass", "mass_ratio"]
                ["chirp_mass", "symmetric_mass_ratio"]
                ["chirp_mass", "mass_ratio", "chi_eff", "chi_a"]
                ["chirp_mass", "symmetric_mass_ratio", "chi_eff", "chi_a"]

        Returns
        -------
        obj: Mass_Spin
            The Mass_Spin object

        """
        obj = cls()
        obj.grid_coordinates = grid_coordinates
        if grid_coordinates[0] == "chirp_mass":
            if grid_coordinates[1] == "mass_ratio":
                obj._mass_1, obj._mass_2 = cls.mcq_to_m1m2(x1, x2)
            elif grid_coordinates[1] == "symmetric_mass_ratio":
                obj._mass_1, obj._mass_2 = cls.mceta_to_m1m2(x1, x2)
            else:
                raise ValueError("Invalid grid_coordinates")
        else:
            raise ValueError("Invalid grid_coordinates")
        return obj

    @classmethod
    def from_m1m2s1zs2z(cls,
                        mass_1,
                        mass_2,
                        spin_1z,
                        spin_2z,
                        grid_coordinates=None):
        obj = cls()
        obj._mass_1 = mass_1
        obj._mass_2 = mass_2
        obj._spin_1z = spin_1z
        obj._spin_2z = spin_2z
        obj.grid_coordinates = grid_coordinates
        return obj

    @classmethod
    def from_m1m2(cls, mass_1, mass_2, grid_coordinates=None):
        obj = cls()
        obj._mass_1 = mass_1
        obj._mass_2 = mass_2
        obj.grid_coordinates = grid_coordinates
        return obj

    @classmethod
    def from_mceta(cls, mc, eta, grid_coordinates=None):
        obj = cls()
        m1, m2 = cls.mceta_to_m1m2(mc, eta)
        obj._mass_1 = m1
        obj._mass_2 = m2
        obj.grid_coordinates = grid_coordinates
        return obj

    @classmethod
    def from_mcq(cls, mc, q, grid_coordinates=None):
        obj = cls()
        m1, m2 = cls.mcq_to_m1m2(mc, q)
        obj._mass_1 = m1
        obj._mass_2 = m2
        obj.grid_coordinates = grid_coordinates
        return obj

    @property
    def x1x2(self):
        x1 = self[self.grid_coordinates[0]]
        x2 = self[self.grid_coordinates[1]]
        return x1, x2

    @property
    def mass_1(self):
        return self._mass_1

    @mass_1.setter
    def mass_1(self, value):
        self._mass_1 = value

    @property
    def mass_2(self):
        return self._mass_1

    @mass_2.setter
    def mass_2(self, value):
        self._mass_1 = value

    @property
    def chirp_mass(self):
        chirp_mass = self.m1m2_to_mc(self._mass_1, self._mass_2)
        return chirp_mass

    @property
    def mass_ratio(self):
        mass_ratio = self.m1m2_to_q(self._mass_1, self._mass_2)
        return mass_ratio

    @property
    def symmetric_mass_ratio(self):
        symmetric_mass_ratio = self.m1m2_to_eta(self._mass_1, self._mass_2)
        return symmetric_mass_ratio

    @property
    def mceta(self):
        return self.chirp_mass, self.symmetric_mass_ratio

    @property
    def m1m2(self):
        return self._mass_1, self._mass_2

    @property
    def mcq(self):
        return self.chirp_mass, self.mass_ratio

    @classmethod
    def _norm_sym_ratio(cls, eta):
        """
        Copied from https://git.ligo.org/rapidpe-rift/rapidpe/-/blob/master/rapid_pe/lalsimutils.py
        """  # noqa E501
        # Assert phyisicality
        assert np.all(eta <= 0.25)
        return np.sqrt(1. - 4. * eta)

    @classmethod
    def mceta_to_m1m2(cls, mc, eta):
        """
        Compute component masses from Mc, eta. Returns m1 >= m2
        """
        m1 = 0.5*mc*eta**(-0.6)*(1. + cls._norm_sym_ratio(eta))  # 3/5=0.6
        m2 = 0.5*mc*eta**(-0.6)*(1. - cls._norm_sym_ratio(eta))  # 3/5=0.6
        return m1, m2

    @classmethod
    def m1m2_to_mceta(cls, m1, m2):
        """
        Compute chirp mass and symmetric mass ratio from component masses
        """
        mc = cls.m1m2_to_mc(m1, m2)
        eta = cls.m1m2_to_eta(m1, m2)
        return mc, eta

    @classmethod
    def mcq_to_m1m2(cls, mc, q):
        m = mc * (1+q)**0.2  # 1/5=0.2
        m1 = m / q**0.6  # 3/5=0.6
        m2 = m * q**0.4  # 2/5=0.4
        return m1, m2

    @classmethod
    def m1m2_to_mcq(cls, m1, m2):
        """
        Compute chirp mass and mass ratio from component masses

        Note:
            mass ratio q=m2/m1
            forcing m1 >= m2
        """
        mc = cls.m1m2_to_mc(m1, m2)
        q = cls.m1m2_to_q(m1, m2)
        return mc, q

    @classmethod
    def m1m2_to_q(cls, m1, m2):
        """
        Compute mass ratio from component masses
        """
        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        return q

    @classmethod
    def m1m2_to_mc(cls, m1, m2):
        """
        Compute chirp mass from component masses
        """
        mc = (m1*m2)**0.6 * (m1+m2)**(-0.2)  # 3/5=0.6, 1/5=0.2
        return mc

    @classmethod
    def m1m2_to_eta(cls, m1, m2):
        """
        Compute symmetric mass ratio from component masses
        """
        eta = m1*m2 / (m1+m2)**2
        return eta


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
# For legacy support
# =====================================================================


def transform_m1m2_to_mceta(m1, m2):
    return Mass_Spin.m1m2_to_mceta(m1, m2)


def transform_mceta_to_m1m2(mc, eta):
    return Mass_Spin.mceta_to_m1m2(mc, eta)


# =====================================================================
