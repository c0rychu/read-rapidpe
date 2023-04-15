"""
Read and parse RapidPE output result for post-processing.
Author: Cory Chu <cory@gwlab.page>
"""

import re
import numpy as np
from .grid_point import RapidPE_grid_point
from .transform import transform_m1m2_to_mceta, transform_mceta_to_m1m2
from .transform import jacobian_mceta_by_m1m2

from matplotlib.tri import Triangulation
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.stats import multinomial


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
            self.grid_points = []  # FIXME: it's actually a np.array
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
    def from_xml_array(cls,
                       xml_array,
                       use_numpy=True,
                       use_ligolw=True,
                       extrinsic_table=True):
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
            RapidPE_grid_point.from_xml(
                xml_array[0],
                use_numpy=use_numpy,
                use_ligolw=use_ligolw,
                extrinsic_table=extrinsic_table
            ).intrinsic_table.keys()
        result._keys = list(result._keys)

        # Initialize attributes
        for attr in result._keys:
            setattr(result, attr, np.zeros(N))

        # Initialize "grid_points" attribute
        result.grid_points = np.empty(N, dtype=object)

        # Setup for "iteration" attribute
        result._keys.append("iteration")
        result.iteration = np.zeros(N, dtype=int)
        re_iter = re.compile(r"ILE_iteration_([0-9]+).+\.xml\.gz")

        # Read XML grid points
        for i, filename in enumerate(xml_array):
            grid_point = RapidPE_grid_point.from_xml(
                filename,
                use_numpy=use_numpy,
                use_ligolw=use_ligolw,
                extrinsic_table=extrinsic_table
            )

            # Append grid-points
            result.grid_points[i] = grid_point

            # Append Intrinsic Parameters of grid-points
            for attr in result._keys:
                try:
                    getattr(result, attr)[i] = \
                        grid_point.intrinsic_table[attr][0]
                except KeyError:
                    pass

            # Append "iteration", i.e., grid-refinement-level
            result.iteration[i] = int(re_iter.search(filename).group(1))

        # Add chirp_mass and symmetric_mass_ratio
        if ("mass_1" in result._keys) and ("mass_2" in result._keys):
            result.chirp_mass, result.symmetric_mass_ratio = \
                transform_m1m2_to_mceta(result.mass_1, result.mass_2)
            result._keys.extend(["chirp_mass", "symmetric_mass_ratio"])

        return cls(result)

    def do_interpolate_marg_log_likelihood_m1m2(
            self,
            method="linear-scipy",
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

        _supported_methods = \
            'method= "cubic", "linear", "linear-scipy", "nearest-scipy",' \
            '"cubic-scipy", "gaussian", or "gaussian-renormalized"'

        if method == "gaussian" or method == "gaussian-renormalized":
            # def gaussian_log_likelihood(m1, m2):
            #     mc_arr, eta_arr = transform_m1m2_to_mceta(m1, m2)
            #     sigma_mc = grid_separation_min(self.chirp_mass) *\
            #         gaussian_sigma_to_grid_size_ratio
            #     sigma_eta = grid_separation_min(self.symmetric_mass_ratio) *\
            #         gaussian_sigma_to_grid_size_ratio

            #     likelihood = np.zeros_like(mc_arr)
            #     for i in range(len(self.chirp_mass)):
            #         likelihood += \
            #             np.exp(self.marg_log_likelihood[i]) * \
            #             np.exp(
            #                 (-0.5/sigma_mc**2
            #                  * (mc_arr - self.chirp_mass[i])**2) +
            #                 (-0.5/sigma_eta**2
            #                  * (eta_arr - self.symmetric_mass_ratio[i])**2)
            #             )
            #     return np.log(likelihood)
            def gaussian_log_likelihood(m1, m2):
                mc_arr, eta_arr = transform_m1m2_to_mceta(m1, m2)

                grid_levels = np.unique(self.iteration)
                sigma_mc = {}
                sigma_eta = {}

                for gl in grid_levels:
                    sigma_mc[gl] = grid_separation_min(
                            self.chirp_mass[self.iteration == gl]
                        ) * gaussian_sigma_to_grid_size_ratio
                    sigma_eta[gl] = grid_separation_min(
                            self.symmetric_mass_ratio[self.iteration == gl]
                        ) * gaussian_sigma_to_grid_size_ratio

                likelihood = np.zeros_like(mc_arr)
                for i in range(len(self.chirp_mass)):
                    likelihood += \
                        np.exp(self.marg_log_likelihood[i]) * \
                        np.exp(
                            (-0.5/sigma_mc[self.iteration[i]]**2
                             * (mc_arr - self.chirp_mass[i])**2) +
                            (-0.5/sigma_eta[self.iteration[i]]**2
                             * (eta_arr - self.symmetric_mass_ratio[i])**2)
                        )
                return np.log(likelihood)
            self.log_likelihood = gaussian_log_likelihood

            if method == "gaussian-renormalized":
                # FIXME: The max point can shft if sigma is large.

                # old_max = np.exp(self.marg_log_likelihood.max())
                # old_max_idx = np.argmax(self.marg_log_likelihood)
                # new_max = np.exp(
                #     gaussian_log_likelihood(self.mass_1[old_max_idx],
                #                             self.mass_2[old_max_idx])
                # )

                self_interp = gaussian_log_likelihood(self.mass_1, self.mass_2)
                new_max = self_interp.max()
                new_max_idx = np.argmax(self_interp)
                old_max = self.marg_log_likelihood[new_max_idx]

                print("old_max: ", np.exp(old_max))
                print("new_max: ", np.exp(new_max))

                def gaussian_log_likelihood_renormalized(m1, m2):
                    return gaussian_log_likelihood(m1, m2) + old_max - new_max

                self.log_likelihood = gaussian_log_likelihood_renormalized

        elif method == "gaussian-numpy":
            """
            Gaussian approximation using numpy broadcasting.
            FIXME: For some reason, it can be slower than python for-loop...
            """
            def gaussian_log_likelihood(m1, m2):
                result = self
                mc_arr, eta_arr = transform_m1m2_to_mceta(m1, m2)
                sigma_mc = grid_separation_min(result.chirp_mass) *\
                    gaussian_sigma_to_grid_size_ratio
                sigma_eta = grid_separation_min(result.symmetric_mass_ratio) *\
                    gaussian_sigma_to_grid_size_ratio

                likelihood = np.sum(
                    np.exp(result.marg_log_likelihood) *
                    np.exp(
                        (-0.5/sigma_mc**2 *
                            (mc_arr[..., np.newaxis] - result.chirp_mass)**2) +
                        (-0.5/sigma_eta**2 *
                            (eta_arr[..., np.newaxis] -
                                result.symmetric_mass_ratio)**2)
                    ),
                    axis=-1
                )
                return np.log(likelihood)

            self.log_likelihood = gaussian_log_likelihood

        elif "scipy" in method:
            if method == "linear-scipy":
                f = LinearNDInterpolator(
                    list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    self.marg_log_likelihood,
                    rescale=True,
                    fill_value=-100  # FIXME: is -100 okay?
                    )
            elif method == "cubic-scipy":
                f = CloughTocher2DInterpolator(
                    list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    self.marg_log_likelihood,
                    rescale=True,
                    fill_value=-100  # FIXME: is -100 okay?
                    )
            elif method == "nearest-scipy":
                f = NearestNDInterpolator(
                    list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    self.marg_log_likelihood,
                    rescale=True
                    )
            else:
                raise ValueError(_supported_methods)

            def log_likelihood(m1, m2):
                mc, eta = transform_m1m2_to_mceta(m1, m2)
                ll = f(mc, eta)
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
                raise ValueError(_supported_methods)

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

    def generate_samples(self, N=5000, method="gaussian"):
        if method == "gaussian":
            # Grid-level-independent covariance matrix (old method)
            #
            # sigma_mc = grid_separation_min(self.chirp_mass) * 0.5
            # sigma_eta = grid_separation_min(self.symmetric_mass_ratio) * 0.5
            # cov = np.diag([sigma_mc**2, sigma_eta**2])

            # Compute covariance matrix for different grid levels
            grid_levels = np.unique(self.iteration)
            cov = {}
            for gl in grid_levels:
                mask = self.iteration == gl
                sigma_mc = grid_separation_min(
                    self.chirp_mass[mask]) * 0.5
                sigma_eta = grid_separation_min(
                    self.symmetric_mass_ratio[mask]) * 0.5
                cov[gl] = np.diag([sigma_mc**2, sigma_eta**2])

            # Compute likelihood at each grid point
            likelihood = np.exp(self.marg_log_likelihood)
            sum_likelihood = np.sum(likelihood)

            # Compute number of samples for each grid point
            N_multinomial = multinomial(N*20, likelihood/sum_likelihood)
            N_per_grid_point = N_multinomial.rvs(1)[0]

            # Generate samples
            samples = np.zeros([0, 2])
            for mc, eta, lh, gl, n in zip(self.chirp_mass,
                                          self.symmetric_mass_ratio,
                                          likelihood,
                                          self.iteration,
                                          N_per_grid_point):
                samples = np.concatenate([
                    samples,
                    np.random.multivariate_normal([mc, eta], cov[gl], n)
                    ])
            mc, eta = samples.T

            # Mask out the samples outside the grid region
            eta_max = self.symmetric_mass_ratio.max()
            eta_min = self.symmetric_mass_ratio.min()
            mask = np.logical_and(eta <= eta_max, eta >= eta_min)

            # mc_max = self.chirp_mass.max()
            # mc_min = self.chirp_mass.min()
            # mask = np.logical_and(mask, mc >= mc_min)
            # mask = np.logical_and(mask, mc <= mc_max)

            mc, eta = mc[mask], eta[mask]
            m1, m2 = transform_mceta_to_m1m2(mc, eta)

            prob = 1/jacobian_mceta_by_m1m2(m1, m2)
            prob /= np.sum(prob)

            m = [{"m1": m1[i], "m2": m2[i]} for i in range(len(m1))]
            # m_samples = np.random.choice(m, size=N, p=prob)
            m_samples = np.random.choice(m, size=N, p=prob, replace=False)
            m1, m2 = np.array([[x["m1"], x["m2"]] for x in m_samples]).T
            self.samples = {"mass_1": m1, "mass_2": m2}
