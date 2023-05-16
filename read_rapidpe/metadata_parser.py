import ast
import json


def dict_value_to_float(input_dict):
    for key in input_dict.keys():
        input_dict[key] = float(input_dict[key])
    return input_dict


def unpack_dict_list_string(dict_list_string, is_float=False):
    items = ast.literal_eval(dict_list_string)
    items = {item.split("=")[0]: item.split("=")[1] for item in items}
    if is_float:
        dict_value_to_float(items)

    return items


def fix_param_naming(input_dict):
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
