"""
Author: Cory Chu <cory@gwlab.page>
"""

import numpy as np
from .grid_point import RapidPE_grid_point
from .transform import transform_m1m2_to_mceta


class RapidPE_result:
    """
    RapidPE result
    ...

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
        self.__copy__()

    @classmethod
    def from_xml_array(cls, xml_array):
        result = cls()
        N = len(xml_array)
        # Get keys from the 1st xml file's intrinsic_table
        result._keys = \
            RapidPE_grid_point.from_xml(xml_array[0]).intrinsic_table.keys()
        result._keys = list(result._keys)

        for attr in result._keys:
            setattr(result, attr, np.zeros(N))

        for i, filename in enumerate(xml_array):
            grid_point = RapidPE_grid_point.from_xml(filename)

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
