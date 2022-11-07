from ligo.lw import utils, lsctables, ligolw

class RapidPE_XML:
    """
    Parse the RapidPE result ligo-lw XML files

    ...

    Attributes
    ----------
    xmldoc : ligo.lw.ligolw.Document
        The opend ligolw xml file

    intrinsic_table : dict
        Intrinsic parameters get from "sngl_inspiral:table"
        
    extrinsic_table : dict
        Extrinsic parameters get from "sim_inspiral:table"

    """
    def __init__(self, filename: str):
        self.xmldoc = self._open_ligolw(filename)
        self.intrinsic_table = self._get_ligolw_table(lsctables.SnglInspiralTable)
        try:
            self.extrinsic_table = self._get_ligolw_table(lsctables.SimInspiralTable)
        except ValueError as e:
            # if not *samples.xml.gz, there is no "sim_inspiral:table"
            self.extrinsic_table = {}
    
    def _open_ligolw(self, xmlfile: str):
        xmldoc = utils.load_filename(
            xmlfile, contenthandler=ligolw.LIGOLWContentHandler
            )
        return xmldoc

    def _get_ligolw_table(self, lsctable):
        xmltable = lsctable.get_table(self.xmldoc)
        table = { key: [] for key in xmltable.columnnames} 
        for row in xmltable:
            for key in table.keys():
                table[key].append(getattr(row, key))
        return table   