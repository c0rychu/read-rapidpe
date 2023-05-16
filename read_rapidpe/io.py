import ast
import json
import h5py
import numpy as np


def dict_value_to_float(input_dict):
    for key in input_dict.keys():
        input_dict[key] = float(input_dict[key])
    return input_dict


def unpack_dict_list_string(dict_list_string, is_float=False):
    """
    Unpack string of dict list into python dict

    Example:
        '["mass1=1.1","mass2=2.2"]' -> {"mass1": 1.1, "mass2": 2.2}
    """
    items = ast.literal_eval(dict_list_string)
    items = {item.split("=")[0]: item.split("=")[1] for item in items}
    if is_float:
        dict_value_to_float(items)

    return items


def fix_param_naming(input_dict):
    """
    Fix parameter naming of keys in dict to follow pesummary convention

    Example:
        {mass1: 0.5} -> {mass_1: 0.5}
    """
    keymap = {
        "mass1": "mass_1",
        "mass2": "mass_2",
        "spin1z": "spin_1z",
        "spin2z": "spin_2z",
    }
    for old_key in keymap.keys():
        try:
            input_dict[keymap[old_key]] = input_dict.pop(old_key)
        except KeyError:
            pass


def load_event_info_dict_txt(txtfile):
    """
    Load event_info_dict.txt file into python dict
    """
    with open(txtfile, "r") as f:
        event_info = json.load(f)

    for key in ["channel_name", "psd_file"]:
        try:
            event_info[key] = unpack_dict_list_string(event_info[key])
        except KeyError:
            pass

    for key in ["intrinsic_param"]:
        try:
            event_info[key] = \
                 unpack_dict_list_string(event_info[key], is_float=True)
            fix_param_naming(event_info[key])
        except KeyError:
            pass

    for key in ["event_spin"]:
        try:
            fix_param_naming(event_info[key])
        except KeyError:
            pass

    for key in event_info.keys():
        if "time" in key or key in ["snr", "likelihood"]:
            event_info[key] = float(event_info[key])

    return event_info


def load_injection_info_txt(txtfile):
    """
    Load injection_info.txt file into python dict
    """
    with open(txtfile, "r") as f:
        injection_info = json.load(f)
    fix_param_naming(injection_info)
    return injection_info


# ======================
# For HDF5 I/O
# ======================

def dict_of_ndarray_to_recarray(dict_of_ndarray):
    """
    Convert dict of ndarray to numpy record-array
    """
    # return pd.DataFrame(dict_of_ndarray).to_records(index=False)
    keys = dict_of_ndarray.keys()
    names = ", ".join(keys)
    return np.core.records.fromarrays(
        [dict_of_ndarray[key] for key in keys], names=names
        )


def dict_to_hdf_group(input_dict, hdf_group):
    """
    Save nested python dict to existing hdf5 group
    """
    for key, val in input_dict.items():
        if isinstance(val, dict):
            g = hdf_group.create_group(key)
            dict_to_hdf_group(val, g)
        else:
            hdf_group.create_dataset(key, data=val)


def dict_from_hdf_group(hdf_group):
    """
    Recursively load hdf5 group into nested python dict
    """
    output_dict = {}
    for key, val in hdf_group.items():
        # if isinstance(val, h5py._hl.group.Group):
        if isinstance(val, h5py.Group):
            output_dict[key] = dict_from_hdf_group(val)
        else:
            val = val[()]
            if isinstance(val, bytes):
                val = val.decode()
            output_dict[key] = val

    return output_dict
