"""
Author: Cory Chu <cory@gwlab.page>
"""

import numpy as np
from .grid_point import RapidPE_grid_point
from .transform import transform_m1m2_to_mceta

from matplotlib.tri import Triangulation
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator
from scipy.interpolate import LinearNDInterpolator


def unique_with_tolerance(array, tolerance):
    tolerance = np.abs(tolerance)
    sorted_array = np.sort(array)
    diff = np.diff(sorted_array)
    mask = np.append(True, np.where(diff < tolerance, False, True))
    return sorted_array[mask]


def grid_separation_min(array):
    tolerance = (array.max() - array.min()) / np.sqrt(len(array)) / 200
    # FIXME: is 200 okay?
    # FIXME: is np.sqrt(len(array)) okay?
    x = unique_with_tolerance(array, tolerance=tolerance)
    return np.diff(x).min()


class RapidPE_result:
    """
    RapidPE Result

    It can be created from a list of *.xml.gz files using
        RapidPE_result.from_xml_array([a.xml.gz, b.xml.gz, ...])
    Check help(RapidPE_result.from_xml_array) for an example.

    ...

    Attributes
    ----------
    grid_points : [RapidPE_grid_point]
        list of all grid points data

    mass_i : numpy.ndarray
        array of mass_i over grid points

    spin_iz : numpy.ndarray
        array of spin_iz over grid points

    Similar cases for chirp_mass, symmetric_mass_ratio, ...

    Methods
    -------
    from_xml_array([a.xml.gz, b.xml.gz, ...]):
        Get result from xml.gz files

    do_interpolate_marg_log_likelihood_m1m2(method="cubic"):
        Perfom interpolation of marg_log_likelihood
        After executing, a new method log_likelihood(m1, m2)
        will be created.

    log_likelihood(m1, m2):
        Interpolated log_likelihood

    """

    def __init__(self, result=None):
        if result is None:
            self.grid_points = []
            self._keys = []
        else:
            self.grid_points = result.grid_points
            self._keys = result._keys

            for attr in result._keys:
                try:
                    setattr(self, attr, getattr(result, attr))
                except AttributeError:
                    pass
            # Try:
            # self.mass_1 = result.mass_1
            # self.mass_2 = result.mass_2
            # self.spin_1z = result.spin_1z
            # self.spin_2z = result.spin_2z
            # self.marg_log_likelihood = result.marg_log_likelihood
            # ...

    def __copy__(self):
        return RapidPE_result(self)

    def copy(self):
        return self.__copy__()

    @classmethod
    def from_xml_array(cls, xml_array, use_numpy=True):
        """
        Get result from xml.gz files

        Example:
            import glob
            results_dir = "path/to/results"
            result_xml_files = glob.glob(results_dir+"*.xml.gz")
            result = RapidPE_result.from_xml_array(result_xml_files)
        """
        result = cls()
        N = len(xml_array)
        # Get keys from the 1st xml file's intrinsic_table
        result._keys = \
            RapidPE_grid_point.from_xml(xml_array[0]).intrinsic_table.keys()
        result._keys = list(result._keys)

        for attr in result._keys:
            setattr(result, attr, np.zeros(N))

        for i, filename in enumerate(xml_array):
            grid_point = RapidPE_grid_point.from_xml(filename, use_numpy)

            # Append grid-points
            result.grid_points.append(grid_point)

            # Append Intrinsic Parameters of grid-points
            for attr in result._keys:
                try:
                    getattr(result, attr)[i] = \
                        grid_point.intrinsic_table[attr][0]
                except KeyError:
                    pass

        if ("mass_1" in result._keys) and ("mass_2" in result._keys):
            result.chirp_mass, result.symmetric_mass_ratio = \
                transform_m1m2_to_mceta(result.mass_1, result.mass_2)
            result._keys.extend(["chirp_mass", "symmetric_mass_ratio"])

        return cls(result)

    def do_interpolate_marg_log_likelihood_m1m2(
            self,
            method="cubic",
            gaussian_sigma_to_grid_size_ratio=0.5
            ):
        """
        Perfom triangular interpolation of marg_log_likelihood in
        chirp_mass (M_c), symmetric_mass_ratio (eta) space.
        After executing, a new method log_likelihood(m1, m2)
        will be created.

        Parameters
        ----------
            method: str
                currently, we support "cubic", "linear",
                "linear-scipy", and "gaussian".

            gaussian_sigma_to_grid_size_ratio=0.5: float
                if using method="gaussian", we can change the
                sigma of gaussian with respect to grid size

        """
        if method == "gaussian":
            def gaussian_log_likelihood(m1, m2):
                result = self
                mc_arr, eta_arr = transform_m1m2_to_mceta(m1, m2)
                sigma_mc = grid_separation_min(result.chirp_mass) *\
                    gaussian_sigma_to_grid_size_ratio
                sigma_eta = grid_separation_min(result.symmetric_mass_ratio) *\
                    gaussian_sigma_to_grid_size_ratio

                likelihood = np.zeros_like(mc_arr)
                for i in range(len(result.chirp_mass)):
                    likelihood += \
                        np.exp(result.marg_log_likelihood[i]) * \
                        np.exp(
                            (-0.5/sigma_mc**2
                             * (mc_arr - result.chirp_mass[i])**2) +
                            (-0.5/sigma_eta**2
                             * (eta_arr - result.symmetric_mass_ratio[i])**2)
                        )
                # print(sigma_eta, sigma_mc)
                return np.log(likelihood)

            self.log_likelihood = gaussian_log_likelihood
        elif method == "linear-scipy":

            f = LinearNDInterpolator(
                list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                self.marg_log_likelihood,
                fill_value=-100  # FIXME: is -100 okay?
                )

            def log_likelihood(m1, m2):
                mc, eta = transform_m1m2_to_mceta(m1, m2)
                ll = f(mc, eta)
                # ll = np.ma.fix_invalid(ll, fill_value=-100).data
                # FIXME: is -100 okay?
                return ll

            self.log_likelihood = log_likelihood

        else:
            triangles = Triangulation(self.chirp_mass,
                                      self.symmetric_mass_ratio)

            if method == "cubic":
                f = CubicTriInterpolator(triangles, self.marg_log_likelihood)
            elif method == "linear":
                f = LinearTriInterpolator(triangles, self.marg_log_likelihood)
            else:
                raise ValueError(
                    'method= "cubic", "linear", "linear-scipy", or "gaussian"'
                    )

            def log_likelihood(m1, m2):
                # FIXME: if m1, m2 is not numpy.ndarray (e.g, scalar, it fails)
                mc, eta = transform_m1m2_to_mceta(m1, m2)
                ll = f(mc, eta)
                ll = np.ma.fix_invalid(ll, fill_value=-100).data
                # FIXME: is -100 okay?
                return ll

            self.log_likelihood = log_likelihood

    def do_interpolate_marg_log_likelihood_mass_spin():
        # TODO: do 4-D interpolation using
        #       scipy.interpolate.LinearNDInterpolator
        #       Ref:https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html
        raise NotImplementedError("to be support in the future")
