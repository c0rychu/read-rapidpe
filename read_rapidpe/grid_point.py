"""
Read single grid point output from RapidPE output.
Author: Cory Chu <cory@gwlab.page>
"""

import xml.etree.ElementTree as ET
import gzip
import numpy as np
from pathlib import Path

# For ligolw reading
from ligo.lw import utils, lsctables, ligolw
lsctables.use_in(ligolw.LIGOLWContentHandler)


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
            self.xml_filename = ""
        else:
            self.xml_filename = grid_point.xml_filename
            self.intrinsic_table_raw = grid_point.intrinsic_table_raw
            self.extrinsic_table_raw = grid_point.extrinsic_table_raw
            self.intrinsic_table = {}
            self._set_intrinsic_table()
            self.extrinsic_table = {}
            self._set_extrinsic_table()
            self._fix_intrinsic_table_spin()  # a temporary solution

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

    def _fix_intrinsic_table_spin(self):
        """
        Fix the missing spins in intrinsic_table.
        """
        if ("spin_1z" not in self.intrinsic_table) and \
           ("spin_2z" not in self.intrinsic_table) and \
           ("spin_1z" in self.extrinsic_table) and \
           ("spin_2z" in self.extrinsic_table):
            self.intrinsic_table["spin_1z"] = \
                self.extrinsic_table["spin_1z"][0:1]
            self.intrinsic_table["spin_2z"] = \
                self.extrinsic_table["spin_2z"][0:1]

    def _set_extrinsic_table(self):
        # Maps are "rapidpe_name": "canonical_name"
        extrinsic_parameter_map = {
                                    "mass1": "mass_1",
                                    "mass2": "mass_2",
                                    "spin1z": "spin_1z",
                                    "spin2z": "spin_2z",
                                    "distance": "luminosity_distance",
                                    "latitude": "dec",
                                    "longitude": "ra",
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
    def from_xml(cls,
                 filename: str,
                 use_numpy=True,
                 use_ligolw=True,
                 extrinsic_table=True):
        """
        Extract XML, assign to "raw" attributes
        "sngl_inspiral:table" -> self.intrinsic_table_raw
        "sim_inspiral:table" -> self.extrinsic_table_raw
        """

        input_xml_gz = filename

        if use_ligolw:
            # time-consuming, should open only once.
            cls.xmldoc = cls._open_ligolw(input_xml_gz)

        grid_point = cls()
        grid_point.xml_filename = Path(input_xml_gz).stem

        grid_point.intrinsic_table_raw = cls._get_ligolw_table(
            input_xml_gz,
            tablename="sngl_inspiral:table",
            use_numpy=use_numpy,
            use_ligolw=use_ligolw
            )

        if extrinsic_table:
            grid_point.extrinsic_table_raw = cls._get_ligolw_table(
                input_xml_gz,
                tablename="sim_inspiral:table",
                use_numpy=use_numpy,
                use_ligolw=use_ligolw
                )
        return cls(grid_point)

    # ===============
    # Private Methods
    # ===============

    @classmethod
    def _open_ligolw(cls, xmlfile: str):
        xmldoc = utils.load_filename(
            xmlfile, contenthandler=ligolw.LIGOLWContentHandler
            )
        return xmldoc

    @classmethod
    def _get_root_xml_gz(cls, input_xml_gz):
        with gzip.open(input_xml_gz, 'r') as input_xml:
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
    def _append_rows_numpy(cls, keys, rows):
        rows = rows.splitlines()
        N = len(rows)
        result = {key: np.zeros(N) for key in keys}
        for row_idx, row in enumerate(rows):
            row = row.split(',')
            # For some reason, some table have extra trailing ","
            # That makes an extra empty column in each row
            # So, we only keep first len(keys) columns
            row = row[0:len(keys)]
            for i, key in enumerate(keys):
                result[key][row_idx] = float(row[i])
        return result

    @classmethod
    def _get_ligolw_table(
            cls,
            input_xml_gz,
            tablename="sngl_inspiral:table",
            use_numpy=True,
            use_ligolw=True
            ):

        if use_ligolw:
            if tablename == "sngl_inspiral:table":
                lsctable = lsctables.SnglInspiralTable
            elif tablename == "sim_inspiral:table":
                lsctable = lsctables.SimInspiralTable
            else:
                raise ValueError("tablename is not supported.")

            xmltable = lsctable.get_table(cls.xmldoc)
            N = len(xmltable)

            if use_numpy:
                table = {key: np.zeros(N) for key in xmltable.columnnames}
                for row_idx, row in enumerate(xmltable):
                    for key in table.keys():
                        table[key][row_idx] = getattr(row, key)
                return table
            else:
                table = {key: [] for key in xmltable.columnnames}
                for row in xmltable:
                    for key in table.keys():
                        table[key].append(getattr(row, key))
                return table

        else:  # Not use_ligolw (faster)
            # Get XML root
            root = cls._get_root_xml_gz(input_xml_gz)

            # Find inspiral Table
            inspiral_table = root.find("*[@Name='" + tablename + "']")

            # Find coloum names
            keys = [i.attrib["Name"].split(":")[-1]
                    for i in inspiral_table.iter("Column")]

            # Clean up string
            # TODO: Check reliablity! This is a dirty way to clean up the
            #       ligolwXML
            s = inspiral_table.find("Stream").text
            s = s.replace("\t", "")
            if s[0] == "\n":
                s = s[1:]  # Remove the extra "\n" in the begining of Stream

            # Combine key-value
            if use_numpy:
                return cls._append_rows_numpy(keys, s)
            else:
                return cls._append_rows(keys, s)
