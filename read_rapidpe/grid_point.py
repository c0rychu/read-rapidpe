"""
Author: Cory Chu <cory@gwlab.page>
"""

import xml.etree.ElementTree as ET
import gzip


class RapidPE_grid_point:
    """
    RapidPE Grid Point

    ...

    """
    def __init__(self, grid_point=None):
        if grid_point is None:
            self.intrinsic_table = {}
            self.intrinsic_table_raw = {}
            self.extrinsic_table = {}
            self.extrinsic_table_raw = {}
        else:
            self.intrinsic_table_raw = grid_point.intrinsic_table_raw
            self.extrinsic_table_raw = grid_point.extrinsic_table_raw
            self.intrinsic_table = {}
            self._set_intrinsic_table()
            self.extrinsic_table = {}
            self._set_extrinsic_table()

    def _set_intrinsic_table(self):
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
        for key in intrinsic_parameter_map:
            try:
                self.intrinsic_table[intrinsic_parameter_map[key]] \
                    = self.intrinsic_table_raw[key]
            except KeyError:
                pass

    def _set_extrinsic_table(self):
        # Maps are "rapidpe_name": "canonical_name"
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

    @classmethod
    def from_xml(cls, filename: str):
        """
        Extract XML, assign to "raw" attributes
        "sngl_inspiral:table" -> self.intrinsic_table_raw
        "sim_inspiral:table" -> self.extrinsic_table_raw
        """

        input_xml_gz = filename

        grid_point = cls()
        grid_point.intrinsic_table_raw = cls._get_ligolw_table(
            input_xml_gz,
            tablename="sngl_inspiral:table"
            )
        grid_point.extrinsic_table_raw = cls._get_ligolw_table(
            input_xml_gz,
            tablename="sim_inspiral:table"
            )
        return cls(grid_point)

    # ===============
    # Private Methods
    # ===============

    @classmethod
    def _get_root_xml_gz(cls, input_xml_gz):
        input_xml = gzip.open(input_xml_gz, 'r')
        tree = ET.parse(input_xml)
        root = tree.getroot()
        return root

    @classmethod
    def _append_rows(cls, keys, rows):
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

    @classmethod
    def _get_ligolw_table(
            cls,
            input_xml_gz,
            tablename="sngl_inspiral:table"
            ):

        # Get XML root
        root = cls._get_root_xml_gz(input_xml_gz)

        # Find inspiral Table
        inspiral_table = root.find("*[@Name='" + tablename + "']")

        # Find coloum names
        keys = [i.attrib["Name"].split(":")[-1]
                for i in inspiral_table.iter("Column")]

        # Clean up string
        # TODO: Check reliablity! This is a dirty way to clean up the ligolwXML
        s = inspiral_table.find("Stream").text
        s = s.replace("\t", "")
        if s[0] == "\n":
            s = s[1:]  # Remove the extra "\n" in the begining of Stream string

        # Combine key-value
        return cls._append_rows(keys, s)