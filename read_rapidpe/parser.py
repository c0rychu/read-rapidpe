"""
Parse the RapidPE results
Author: Cory Chu <cory@gwlab.page>
"""

# For RapidPE_XML_fast
import xml.etree.ElementTree as ET
import gzip

# For RapidPE_XML
from ligo.lw import utils, lsctables, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)


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
        # "sim_inspiral:table" -> self.extrinsic_table_raw
        self.intrinsic_table_raw = self._get_ligolw_table(
            lsctables.SnglInspiralTable
            )
        try:
            self.extrinsic_table_raw = self._get_ligolw_table(
                lsctables.SimInspiralTable
                )
        except ValueError:
            # if not *samples.xml.gz or new version xml,
            # there is no "sim_inspiral:table"
            self.extrinsic_table_raw = {}
            pass

        # Provide canonical interface with two attributes
        #   self.intrinsic_table
        #   self.extrinsic_table
        # =============================================
        # TODO: Check the naming convention in bilby and PESummary
        #       [bilby]    https://arxiv.org/pdf/2006.00714.pdf
        #       [PESummary]https://docs.ligo.org/lscsoft/pesummary/stable_docs/gw/parameters.html
        #       spin1z = spin_1z(PESummary)? (maybe = chi_1x= spin_1_z (bilby))
        #       spin2z = spin_2z(PESummary)? (maybe = chi_2 = spin_2_z (bilby))
        #       distance = luminosity_distance?
        #       latitude = dec ?
        #       longitude = ra ?
        #       inclination = theta_jn ?
        #       polarization = psi?
        #       sampling_function is
        #         the p_s in Eq. 28 of https://arxiv.org/pdf/1502.04370.pdf
        # =============================================

        # Maps are "rapidpe_name": "canonical_name"
        intrinsic_parameter_map = {
                                    "mass1": "mass_1",
                                    "mass2": "mass_2",
                                    "spin1z": "spin_1z",
                                    "spin2z": "spin_2z",
                                    "snr": "marg_log_likelihood",
                                    "tau0": "tau0",
                                    "tau3": "tau3",
                                    "alpha4": "eccentricity",
                                }
        self.intrinsic_table = {}
        for key in intrinsic_parameter_map:
            try:
                self.intrinsic_table[intrinsic_parameter_map[key]] \
                    = self.intrinsic_table_raw[key]
            except KeyError:
                pass

        extrinsic_parameter_map = {
                                    "mass1": "mass_1",
                                    "mass2": "mass_2",
                                    "distance": "luminosity_distance",
                                    "latitude": "latitude",
                                    "longitude": "longitude",
                                    # "inclination": "theta_jn",
                                    "inclination": "iota",
                                    "polarization": "psi",
                                    "alpha1": "log_likelihood",
                                    "alpha2": "prior",
                                    "alpha3": "sampling_function",
                                }
        self.extrinsic_table = {}
        for key in extrinsic_parameter_map:
            try:
                self.extrinsic_table[extrinsic_parameter_map[key]] \
                    = self.extrinsic_table_raw[key]
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


class RapidPE_XML_fast:
    """
    Faster Parser than RapidPE_XML

    ...

    """
    def __init__(self, filename: str):

        # Extract XML, assign to "raw" attributes
        # "sngl_inspiral:table" -> self.intrinsic_table_raw
        # "sim_inspiral:table" -> self.extrinsic_table_raw
        input_xml_gz = filename

        self.intrinsic_table_raw = self._get_ligolw_table(
            input_xml_gz,
            tablename="sngl_inspiral:table"
            )
        self.extrinsic_table_raw = self._get_ligolw_table(
            input_xml_gz,
            tablename="sim_inspiral:table"
            )

        # Provide canonical interface with two attributes
        #   self.intrinsic_table
        #   self.extrinsic_table

        # Maps are "rapidpe_name": "canonical_name"
        intrinsic_parameter_map = {
                                    "mass1": "mass_1",
                                    "mass2": "mass_2",
                                    "spin1z": "spin_1z",
                                    "spin2z": "spin_2z",
                                    "snr": "marg_log_likelihood",
                                    "tau0": "tau0",
                                    "tau3": "tau3",
                                    "alpha4": "eccentricity",
                                }
        self.intrinsic_table = {}
        for key in intrinsic_parameter_map:
            try:
                self.intrinsic_table[intrinsic_parameter_map[key]] \
                    = self.intrinsic_table_raw[key]
            except KeyError:
                pass

        extrinsic_parameter_map = {
                                    "mass1": "mass_1",
                                    "mass2": "mass_2",
                                    "distance": "luminosity_distance",
                                    "latitude": "latitude",
                                    "longitude": "longitude",
                                    # "inclination": "theta_jn",
                                    "inclination": "iota",
                                    "polarization": "psi",
                                    "alpha1": "log_likelihood",
                                    "alpha2": "prior",
                                    "alpha3": "sampling_function",
                                }
        self.extrinsic_table = {}
        for key in extrinsic_parameter_map:
            try:
                self.extrinsic_table[extrinsic_parameter_map[key]] \
                    = self.extrinsic_table_raw[key]
            except KeyError:
                pass

    # Private Methods
    def _get_root_xml_gz(self, input_xml_gz):
        input_xml = gzip.open(input_xml_gz, 'r')
        tree = ET.parse(input_xml)
        root = tree.getroot()
        return root

    def _append_rows(self, keys, rows):
        result = {key: [] for key in keys}
        for row in rows.splitlines():
            row = row.split(',')
            # For some reason, some table have extra trailing ","
            # That makes an extra empty column in each row
            # So, we only keep first len(keys) columns
            row = row[0:len(keys)]
            for i, key in enumerate(keys):
                result[key].append(float(row[i]))
        return result

    def _get_ligolw_table(
            self,
            input_xml_gz,
            tablename="sngl_inspiral:table"
            ):

        # Get XML root
        root = self._get_root_xml_gz(input_xml_gz)

        # Find inspiral Table
        inspiral_table = root.find("*[@Name='" + tablename + "']")

        # Find coloum names
        keys = [i.attrib["Name"].split(":")[-1]
                for i in inspiral_table.iter("Column")]

        # Clean up string
        # TODO: Check reliablity! This is a dirty way to clean up the ligolwXML
        s = inspiral_table.find("Stream").text
        s = s.replace("\t", "")
        s = s[1:]  # Remove the extra "\n" in the begining of Stream string

        # Combine key-value
        return self._append_rows(keys, s)
