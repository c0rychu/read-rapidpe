"""
Read and parse RapidPE output result for post-processing.
Author: Cory Chu <cory@gwlab.page>
"""

from functools import cached_property
from pathlib import Path
from joblib import Parallel, delayed
from configparser import ConfigParser
import re
import numpy as np
import h5py
# import pandas as pd
from .grid_point import RapidPE_grid_point
from .io import load_event_info_dict_txt
from .io import load_injection_info_txt
from .io import dict_of_ndarray_to_recarray
from .io import recarray_to_dict_of_ndarray
from .io import dict_from_hdf_group
from .io import dict_to_hdf_group

from .transform import transform_m1m2_to_mceta, transform_mceta_to_m1m2
from .transform import jacobian_mceta_by_m1m2
from .transform import Mass_Spin

from matplotlib.tri import Triangulation
from matplotlib.tri import LinearTriInterpolator, CubicTriInterpolator
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.stats import multinomial
from scipy.special import logsumexp


# import time  # for profiling


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
            self.grid_points = np.empty(0, dtype=object)
            self._keys = []
        else:
            self.grid_points = result.grid_points
            self._keys = result._keys

            for attr in result._keys + ["event_info",
                                        "injection_info",
                                        "config_info",
                                        "posterior_samples"]:
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

    def __getitem__(self, key):
        """
        Get res.attr by res["attr"]
        """
        return getattr(self, key)

    @cached_property
    def _x(self):
        x = Mass_Spin.from_m1m2(self.mass_1,
                                self.mass_2,
                                grid_coordinates=self.grid_coordinates)
        return x

    @cached_property
    def x1(self):
        return self._x.x1

    @cached_property
    def x2(self):
        return self._x.x2

    @property
    def grid_coordinates(self):
        """
        List of grid coordinates in wich the grid points are rectilinear
        """
        iparam_search = self.config_info["intrinsic_param_to_search"]
        distant_coord = self.config_info["distance_coordinates"]
        if iparam_search == "[mass1,mass2]":
            if distant_coord == "mchirp_q":
                return ["chirp_mass", "mass_ratio"]
            elif distant_coord == "mchirp_eta":
                return ["chirp_mass", "symmetric_mass_ratio"]
            else:
                raise NotImplementedError("distance_coordinates not supported")
        elif iparam_search == "[mass1,mass2,spin1z,spin2z]":
            if distant_coord == "mchirp_q":
                return ["chirp_mass", "mass_ratio", "chi_eff", "chi_a"]
            elif distant_coord == "mchirp_eta":
                return ["chirp_mass", "symmetric_mass_ratio", "chi_eff", "chi_a"]  # noqa E501
            else:
                raise NotImplementedError("distance_coordinates not supported")
        else:
            raise NotImplementedError("intrinsic_param_to_search not supported")  # noqa E501

    @cached_property
    def intrinsic_table(self):
        """
        Combine intrinsic tables to a single dict
        """
        # return pd.DataFrame({key: getattr(self, key) for key in self._keys})
        return {key: getattr(self, key) for key in self._keys}

    @cached_property
    def extrinsic_samples(self):
        """
        Combine extrinsic samples to a single dict
        """
        # return pd.concat(
        #      [pd.DataFrame(gp.extrinsic_table) for gp in self.grid_points])
        gps = self.grid_points
        keys_ext = gps[0].extrinsic_table.keys()
        extrinsic_samples = {}
        for key in keys_ext:
            extrinsic_samples[key] = \
                np.concatenate([gp.extrinsic_table[key] for gp in gps])
        return extrinsic_samples

    def __copy__(self):
        return RapidPE_result(self)

    def copy(self):
        return self.__copy__()

    @classmethod
    def from_hdf(cls, hdf_file, extrinsic_table=True):
        """
        Get result from a Rapid-PE HDF file

        Example
        -------
            result = RapidPE_result.from_hdf("path/to/hdf_file")

        Attributes
        ----------
        hdf_file : string
            The path to Rapid-PE HDF file

        extrinsic_table : bool
            Whether loading extrinsic_table as well

        """
        result = cls()
        with h5py.File(hdf_file, "r") as f:
            try:
                # Load grid_points
                gps = f["grid_points"]
                N = len(gps)
                result.grid_points = np.empty(N, dtype=object)
                for i, gp in enumerate(gps.values()):
                    result.grid_points[i] = \
                        RapidPE_grid_point.from_hdf_grid_point_group(
                            hdf_gp_group=gp,
                            extrinsic_table=extrinsic_table
                            )

                # Load intrinsic_table
                it = f["intrinsic_table"]
                result.intrinsic_table = {key: it[key] for key in it.dtype.names}  # noqa E501
                result._keys = list(it.dtype.names)
            except KeyError:
                pass

            # Load other attributes
            for attr in result._keys:
                try:
                    setattr(result, attr, result.intrinsic_table[attr])
                except KeyError:
                    pass

            # Load event_info and injection_info
            for attr in ["event_info", "injection_info", "config_info"]:
                try:
                    x = dict_from_hdf_group(f[attr])
                    setattr(result, attr, x)
                except KeyError:
                    pass

            # Load posterior samples
            try:
                result.posterior_samples = \
                     recarray_to_dict_of_ndarray(f["posterior_samples"])
            except KeyError:
                pass

        return cls(result)

    @classmethod
    def from_run_dir(cls,
                     run_dir,
                     use_numpy=True,
                     use_ligolw=True,
                     extrinsic_table=True,
                     load_results=True,
                     parallel_n=1):
        """
        Get result from a Rapid-PE run_dir

        Example
        -------
            result = RapidPE_result.from_run_dir("path/to/run_dir")

        Attributes
        ----------
        run_dir : string or pathlib's Path() object
            The path to rapid-pe run_dir

        use_ligolw : bool
            Whether using ligo.lw to read xml files

        extrinsic_table : bool
            Whether loading extrinsic_table as well

        """
        run_dir = Path(run_dir)

        if load_results:
            results_dir = Path(run_dir)/Path("results")
            xml_array = [f.as_posix() for f in sorted(results_dir.glob("*.xml.gz"))]  # noqa: E501

            result = cls.from_xml_array(xml_array,
                                        use_numpy=use_numpy,
                                        use_ligolw=use_ligolw,
                                        extrinsic_table=extrinsic_table,
                                        parallel_n=parallel_n)
        else:
            result = cls()

        event_info_dict_txt = run_dir / "event_info_dict.txt"
        injection_info_txt = run_dir / "injection_info.txt"
        config_ini = run_dir/"Config.ini"

        try:
            result.event_info = load_event_info_dict_txt(event_info_dict_txt)
        except FileNotFoundError:
            pass

        try:
            result.injection_info = load_injection_info_txt(injection_info_txt)
        except FileNotFoundError:
            pass

        try:
            config = ConfigParser()
            config.read(config_ini)
            result.config_info = \
                {"distance_coordinates":
                    config["GridRefine"]["distance-coordinates"],
                 "intrinsic_param_to_search":
                    config["General"]["intrinsic_param_to_search"]}
        except KeyError:
            pass

        return result

    @classmethod
    def from_xml_array(cls,
                       xml_array,
                       use_numpy=True,
                       use_ligolw=True,
                       extrinsic_table=True,
                       parallel_n=1):
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
        if parallel_n == 1:
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

        # Parallel Reading XML files
        else:
            # t0 = time.time()
            grid_points = Parallel(n_jobs=parallel_n)(
                delayed(RapidPE_grid_point.from_xml)(
                    filename,
                    use_numpy=use_numpy,
                    use_ligolw=use_ligolw,
                    extrinsic_table=extrinsic_table
                ) for filename in xml_array
                )
            # t1 = time.time()
            for i, grid_point in enumerate(grid_points):
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
                filename = xml_array[i]
                result.iteration[i] = int(re_iter.search(filename).group(1))
            # t2 = time.time()
            # print("parallel reading time: ", t1-t0)
            # print("post-proccessing time: ", t2-t1)

        # Add chirp_mass and symmetric_mass_ratio
        if ("mass_1" in result._keys) and ("mass_2" in result._keys):
            x = Mass_Spin.from_m1m2(result.mass_1, result.mass_2)
            result.chirp_mass = x.chirp_mass
            result.symmetric_mass_ratio = x.symmetric_mass_ratio
            result.mass_ratio = x.mass_ratio
            result._keys.extend(
                ["chirp_mass", "symmetric_mass_ratio", "mass_ratio"])

        return cls(result)

    def to_hdf(self, hdf_filename, extrinsic_table=True, compression=None):
        """
        Save result to hdf file

        Parameters
        ----------
        filename : str
            The name of the hdf file

        compression : str, optional (default: None)
            The compression method, e.g., "gzip", "lzf".
        """

        with h5py.File(hdf_filename, 'w', track_order=True) as f:

            # Check if there are results
            if len(self.grid_points) != 0:

                # Check if there is extrinsic_table
                gp = self.grid_points[0]
                extrinsic_table &= len(gp.extrinsic_table) > 0

                # Create "grid_points" group to hold self.grid_points
                group_grid_points_raw = \
                    f.create_group("grid_points", track_order=True)

                for i, gp in enumerate(self.grid_points):
                    group_gp = group_grid_points_raw.create_group(str(i))

                    # Add intrinsic_table
                    it = dict_of_ndarray_to_recarray(gp.intrinsic_table)
                    group_gp.create_dataset("intrinsic_table", data=it)

                    # Add extrinsic_table
                    if extrinsic_table:
                        et = dict_of_ndarray_to_recarray(gp.extrinsic_table)
                        group_gp.create_dataset("extrinsic_table",
                                                data=et,
                                                compression=compression)

                    # Add xml_filename
                    group_gp.attrs["xml_filename"] = gp.xml_filename

                # Combine intrinsic parameters into "intrinsic_table" dataset
                result_np = dict_of_ndarray_to_recarray(self.intrinsic_table)
                f.create_dataset("intrinsic_table", data=result_np)

                # Create virtual dataset for "extrinsic_samples"
                if extrinsic_table:
                    gps = f["grid_points"]
                    ds0 = f["grid_points/0/extrinsic_table"]
                    n_samples = np.array(
                        [gp["extrinsic_table"].shape[0] for gp in gps.values()])  # noqa: E501
                    n_samples = np.cumsum(n_samples)
                    n_samples = np.concatenate([[0], n_samples])
                    layout = h5py.VirtualLayout(shape=(n_samples[-1], ),
                                                dtype=ds0.dtype)

                    for i, gp in enumerate(gps.values()):
                        vsource = h5py.VirtualSource(gp["extrinsic_table"])
                        layout[n_samples[i]:n_samples[i+1]] = vsource

                    f.create_virtual_dataset('extrinsic_samples', layout)

            # Save event_info, injection_info, and config_info
            for attr in ["event_info", "injection_info", "config_info"]:
                try:
                    x = getattr(self, attr)
                    g = f.create_group(attr)
                    dict_to_hdf_group(x, g)
                except AttributeError:
                    pass

            # Save posterior_samples
            try:
                data = dict_of_ndarray_to_recarray(self.posterior_samples)
                f.create_dataset("posterior_samples", data=data)
            except AttributeError:
                pass

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

        grid_coord_1 = self[self.grid_coordinates[0]]
        grid_coord_2 = self[self.grid_coordinates[1]]

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
                x = Mass_Spin.from_m1m2(m1, m2)
                x1 = x[self.grid_coordinates[0]]
                x2 = x[self.grid_coordinates[1]]

                grid_levels = np.unique(self.iteration)
                sigma_1 = {}
                sigma_2 = {}

                for gl in grid_levels:
                    sigma_1[gl] = grid_separation_min(
                            grid_coord_1[self.iteration == gl]
                        ) * gaussian_sigma_to_grid_size_ratio
                    sigma_2[gl] = grid_separation_min(
                            grid_coord_2[self.iteration == gl]
                        ) * gaussian_sigma_to_grid_size_ratio

                likelihood = np.zeros_like(x1)
                for i in range(len(grid_coord_1)):
                    likelihood += \
                        np.exp(self.marg_log_likelihood[i]) * \
                        np.exp(
                            (-0.5/sigma_1[self.iteration[i]]**2
                             * (x1 - grid_coord_1[i])**2) +
                            (-0.5/sigma_2[self.iteration[i]]**2
                             * (x2 - grid_coord_2[i])**2)
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
                    # list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    list(zip(grid_coord_1, grid_coord_2)),
                    self.marg_log_likelihood,
                    rescale=True,
                    fill_value=-100  # FIXME: is -100 okay?
                    )
            elif method == "cubic-scipy":
                f = CloughTocher2DInterpolator(
                    # list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    list(zip(grid_coord_1, grid_coord_2)),
                    self.marg_log_likelihood,
                    rescale=True,
                    fill_value=-100  # FIXME: is -100 okay?
                    )
            elif method == "nearest-scipy":
                f = NearestNDInterpolator(
                    # list(zip(self.chirp_mass, self.symmetric_mass_ratio)),
                    list(zip(grid_coord_1, grid_coord_2)),
                    self.marg_log_likelihood,
                    rescale=True
                    )
            else:
                raise ValueError(_supported_methods)

            def log_likelihood(m1, m2):
                # mc, eta = transform_m1m2_to_mceta(m1, m2)
                # ll = f(mc, eta)
                x = Mass_Spin.from_m1m2(m1, m2)
                x1 = x[self.grid_coordinates[0]]
                x2 = x[self.grid_coordinates[1]]
                ll = f(x1, x2)
                return ll

            self.log_likelihood = log_likelihood

        else:
            raise DeprecationWarning(_supported_methods)
            # TODO: remove methods uisng matplotlib triangulation

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

            # FIXME: use grid_coordinates instead of hard-coded
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

    def generate_posterior_samples(self,
                                   N=5000,
                                   method="gaussian-resample",
                                   seed=None,
                                   gaussian_sigma_to_grid_size_ratio=1.0,
                                   em_bright_compatible=True):
        """
        Generate posterior samples.
        The generated samples are saved in self.samples

        Parameters
        ----------
        N: int
            Number of samples to generate
            Default: 5000

        method: str
            Method to generate samples
            Default: "gaussian"

        gaussian_sigma_to_grid_size_ratio: float
            Ratio of sigma to grid size
            Default: 1.0

        em_bright_compatible: bool
            If True, generate samples compatible with EM_BRIGHT
            Default: True

        """

        rng = np.random.default_rng(seed)

        if method == "gaussian" or method == "gaussian-resample":
            grid_levels = np.unique(self.iteration)
            cov = {}
            for gl in grid_levels:
                mask = self.iteration == gl
                sigma_x1 = grid_separation_min(
                    self.x1[mask]) * gaussian_sigma_to_grid_size_ratio
                sigma_x2 = grid_separation_min(
                    self.x2[mask]) * gaussian_sigma_to_grid_size_ratio
                cov[gl] = np.diag([sigma_x1**2, sigma_x2**2])

            # Compute normalized relative likelihood at each grid point
            prob = np.exp(self.marg_log_likelihood
                          - logsumexp(self.marg_log_likelihood))

            if method == "gaussian":
                x = Mass_Spin.from_m1m2(self.mass_1,
                                        self.mass_2,
                                        grid_coordinates=self.grid_coordinates)
                prob *= x.jacobian_m1m2_by_x1x2
                prob /= np.sum(prob)

            # Compute number of samples for each grid point
            N_per_grid_point = rng.multinomial(N*20, pvals=prob)

            # Generate samples
            samples = np.zeros([0, 2])
            for x1, x2, gl, n in zip(self.x1,
                                     self.x2,
                                     self.iteration,
                                     N_per_grid_point):
                samples = np.concatenate([
                    samples,
                    rng.multivariate_normal([x1, x2], cov[gl], n)
                    ])
            x1, x2 = samples.T

            # Mask out the samples outside the region of interest
            if self.grid_coordinates[0] == "chirp_mass":
                x1_min = 0.0
            else:
                raise ValueError("Unknown grid coordinate x1")

            if self.grid_coordinates[1] == "mass_ratio":
                x2_max = 1.0
                x2_min = 0.0
            elif self.grid_coordinates[1] == "symmetric_mass_ratio":
                x2_max = 0.25
                x2_min = 0.0
            else:
                raise ValueError("Unknown grid coordinate x2")

            mask = x2 <= x2_max
            mask &= x2 > x2_min
            mask &= x1 > x1_min

            x1, x2 = x1[mask], x2[mask]

            x = Mass_Spin.from_x1x2(x1,
                                    x2,
                                    grid_coordinates=self.grid_coordinates)

            # An ad-hoc cut on the mass_1 and mass_2
            mask_m1m2 = x.mass_1 < 500
            mask_m1m2 &= x.mass_2 > 0.1

            # Re-weight the samples according to the Jacobian such that
            # it has a uniform prior in m1-m2 space
            # Reference: https://dcc.ligo.org/LIGO-T2300198
            if method == "gaussian-resample":
                weight = x.jacobian_m1m2_by_x1x2[mask_m1m2]
                weight /= np.sum(weight)
            else:
                weight = None

            m = {"mass_1": x.mass_1[mask_m1m2],
                 "mass_2": x.mass_2[mask_m1m2]}

            m = dict_of_ndarray_to_recarray(m)
            samples = rng.choice(m, size=N, p=weight, replace=False)
            samples = recarray_to_dict_of_ndarray(samples)

            if em_bright_compatible:
                shape = samples["mass_1"].shape
                event_spin = self.event_info["event_spin"]
                spin_1z = np.broadcast_to(event_spin["spin_1z"], shape)
                spin_2z = np.broadcast_to(event_spin["spin_2z"], shape)

                samples["mass_1_source"] = samples["mass_1"]
                samples["mass_2_source"] = samples["mass_2"]
                samples["spin_1z"] = spin_1z
                samples["spin_2z"] = spin_2z

                self.posterior_samples = samples
            else:
                self.posterior_samples = samples

    def plot_grid(self,
                  posterior_samples=True,
                  true_params=True,
                  legend_loc=None):
        import matplotlib.pyplot as plt
        from .plot import plot_grid

        plot_grid(self, posterior_samples=posterior_samples)

        if true_params:
            legend = False
            try:
                x_inj = Mass_Spin.from_m1m2(
                    self.injection_info["mass_1"],
                    self.injection_info["mass_2"],
                    grid_coordinates=self.grid_coordinates
                )
                plt.scatter(x_inj.x1, x_inj.x2, marker="*", c="r", s=35,
                            label="Injection")
                legend = True
            except AttributeError:
                pass

            try:
                x_pipe = Mass_Spin.from_m1m2(
                    self.event_info["intrinsic_param"]["mass_1"],
                    self.event_info["intrinsic_param"]["mass_2"],
                    grid_coordinates=self.grid_coordinates
                )
                plt.scatter(x_pipe.x1, x_pipe.x2, marker="o", c="b", s=15,
                            label="Search Pipeline")
                legend = True
            except AttributeError:
                pass

            if legend:
                plt.legend(loc=legend_loc, fontsize="8", fancybox=False)

    def plot_corner(self,
                    columns=["mass_1", "mass_2"],
                    title=None,
                    true_params=True,
                    figsize=None,
                    **kwargs):
        from .plot import plot_corner
        from matplotlib.lines import Line2D
        import corner

        samples = {}
        for col in columns:
            try:
                samples[col] = self.posterior_samples[col]
            except KeyError:
                pass

        fig = plot_corner(samples, **kwargs)

        if true_params:
            legend = False
            legend_elements = []
            inj = False
            pipe = False

            x_inj = []
            for col in columns:
                try:
                    x_inj.append(self.injection_info[col])
                    legend = True
                    inj = True
                except (KeyError, AttributeError):
                    x_inj.append(None)
            if inj:
                corner.overplot_lines(fig, x_inj, color="r")
                corner.overplot_points(fig,
                                       [x_inj],
                                       marker="*",
                                       markersize=10,
                                       color="r",
                                       label="Injection")
                legend_elements.append(
                    Line2D([0], [0], marker="*", color="r", markersize=10, label="Injection")  # noqa: E501
                )

            x_pipe = []
            for col in columns:
                try:
                    x_pipe.append(self.event_info["intrinsic_param"][col])
                    legend = True
                    pipe = True
                except (KeyError, AttributeError):
                    x_pipe.append(None)
            if pipe:
                corner.overplot_lines(fig, x_pipe, color="b")
                corner.overplot_points(fig,
                                       [x_pipe],
                                       marker="o",
                                       color="b",
                                       label="Search Pipeline")
                legend_elements.append(
                    Line2D([0], [0], marker="o", color="b", label="Search Pipeline")  # noqa: E501
                )

            if legend:
                fig.legend(handles=legend_elements,
                           bbox_to_anchor=(0.93, 0.85))

        if figsize is not None:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])

        if title is not None:
            fig.suptitle(title, y=1.05)

        return fig
