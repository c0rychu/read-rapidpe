"""
Parse the RapidPE results
Author: Cory Chu <cory@gwlab.page>
"""

from ligo.lw import utils, lsctables, ligolw


class RapidPE_XML:
    """
    Parse the RapidPE result ligo-lw XML files

    ...

    Attributes
    ----------
    TODO: Complete the example/explanation of parameters
    intrinsic_table : dict
        Intrinsic parameters get from "sngl_inspiral:table"
        with canonical parameter names.

        An example:
        {
            "mass_1": [1.4],
            "mass_2": [2.8],
            ...
        }

        Parameters:
        mass_i:
            component masses (in solar mass?)

        spin_iz:
            z-component of spin (what coordinate?)

        marg_log_likelihood:
            natural log of the likelihood
            marginalized over extrinsic parameters

        tau0, tau3:
            chirptime parameters
            It's a kind of mass parameter
            like (m1, m2) or (chirp_mass, mass_ratio)
            See Appendix B of https://arxiv.org/pdf/0706.4437.pdf



    extrinsic_table : dict
        Extrinsic parameters get from "sim_inspiral:table"
        with canonical parameter names


    Attributes (Advanced usage)
    ----------
    xmldoc : ligo.lw.ligolw.Document
        The opened ligolw xml file

    intrinsic_table_raw : dict
        Raw intrinsic parameters get from "sngl_inspiral:table"

    extrinsic_table_raw : dict
        Raw extrinsic parameters get from "sim_inspiral:table"

    """
    def __init__(self, filename: str):
        self.xmldoc = self._open_ligolw(filename)

        # Extract XML, assign to "raw" attributes
        # "sngl_inspiral:table" -> self.intrinsic_table_raw
        # ""sim_inspiral:table" -> self.extrinsic_table_raw
        self.intrinsic_table_raw = self._get_ligolw_table(
            lsctables.SnglInspiralTable
            )
        try:
            self.extrinsic_table_raw = self._get_ligolw_table(
                lsctables.SimInspiralTable
                )
        except ValueError:
            # if not *samples.xml.gz, there is no "sim_inspiral:table"
            self.extrinsic_table_raw = {}
            pass

        # Provide canonical interface with two attributes
        #   self.intrinsic_table
        #   self.extrinsic_table
        # TODO: Check the naming convention in bilby and PESummary
        #       [bilby]    https://arxiv.org/pdf/2006.00714.pdf
        #       [PESummary]https://docs.ligo.org/lscsoft/pesummary/stable_docs/gw/parameters.html
        self.intrinsic_table = {}
        try:
            self.intrinsic_table = {
                "mass_1": self.intrinsic_table_raw["mass1"],
                "mass_2": self.intrinsic_table_raw["mass2"],
                "marg_log_likelihood": self.intrinsic_table_raw["snr"],
                "spin_1z": self.intrinsic_table_raw["spin1z"],
                # spin1z = chi_1(bilby) = spin_1_z(bilby) = spin_1z(PESummary)?
                "spin_2z": self.intrinsic_table_raw["spin2z"],
                # spin2z = chi_2(bilby) = spin_2_z(bilby) = spin_2z(PESummary)?
                "tau0": self.intrinsic_table_raw["tau0"],
                "tau3": self.intrinsic_table_raw["tau3"],
                # (tau0, tau3) are called chirptime parameters
                # It's a kind of mass parameter
                # like (m1, m2) or (chirp_mass, mass_ratio)
                # See Appendix B of https://arxiv.org/pdf/0706.4437.pdf
            }
        except KeyError:
            pass

        self.extrinsic_table = {}
        try:
            self.extrinsic_table = {
                "mass_1": self.extrinsic_table_raw["mass1"],
                "mass_2": self.extrinsic_table_raw["mass2"],
                "luminosity_distance": self.extrinsic_table_raw["distance"],
                # distance = luminosity_distance?
                "latitude": self.extrinsic_table_raw["latitude"],
                # latitude = dec ?
                "longitude": self.extrinsic_table_raw["longitude"],
                # longitude = ra ?
                "theta_jn": self.extrinsic_table_raw["inclination"],
                # inclination = theta_jn ?
                "psi": self.extrinsic_table_raw["polarization"],
                # polarization = psi?
                "log_likelihood": self.extrinsic_table_raw["alpha1"],
                "prior": self.extrinsic_table_raw["alpha2"],
                "sampling_function": self.extrinsic_table_raw["alpha3"],
                # sampling_function is
                # the p_s in Eq. 28 of https://arxiv.org/pdf/1502.04370.pdf
                }
        except KeyError:
            pass

    def _open_ligolw(self, xmlfile: str):
        xmldoc = utils.load_filename(
            xmlfile, contenthandler=ligolw.LIGOLWContentHandler
            )
        return xmldoc

    def _get_ligolw_table(self, lsctable):
        xmltable = lsctable.get_table(self.xmldoc)
        table = {key: [] for key in xmltable.columnnames}
        for row in xmltable:
            for key in table.keys():
                table[key].append(getattr(row, key))
        return table
