from ligo.lw import utils, lsctables, ligolw


class RapidPE_XML:
    """
    Parse the RapidPE result ligo-lw XML files

    ...

    Attributes
    ----------
    xmldoc : ligo.lw.ligolw.Document
        The opened ligolw xml file

    intrinsic_table : dict
        Intrinsic parameters get from "sngl_inspiral:table"

    extrinsic_table : dict
        Extrinsic parameters get from "sim_inspiral:table"

    """
    def __init__(self, filename: str):
        self.xmldoc = self._open_ligolw(filename)
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

        self.intrinsic_table = {}
        try:
            self.intrinsic_table = {
                "mass_1": self.intrinsic_table_raw["mass1"],
                "mass_2": self.intrinsic_table_raw["mass2"],
                "marg_ln_likelihood": self.intrinsic_table_raw["snr"],
                "spin1z": self.intrinsic_table_raw["spin1z"],
                "spin2z": self.intrinsic_table_raw["spin2z"],
                "tau0": self.intrinsic_table_raw["tau0"],
                "tau3": self.intrinsic_table_raw["tau3"]
            }
        except KeyError:
            pass

        self.extrinsic_table = {}
        try:
            self.extrinsic_table = {
                "mass_1": self.extrinsic_table_raw["mass1"],
                "mass_2": self.extrinsic_table_raw["mass2"],
                "distance": self.extrinsic_table_raw["distance"],
                "latitude": self.extrinsic_table_raw["latitude"],
                "longitude": self.extrinsic_table_raw["longitude"],
                "inclination": self.extrinsic_table_raw["inclination"],
                "polarization": self.extrinsic_table_raw["polarization"],
                "ln_likelihood": self.extrinsic_table_raw["alpha1"],
                "prior": self.extrinsic_table_raw["alpha2"],
                "sampling_function": self.extrinsic_table_raw["alpha3"],
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
