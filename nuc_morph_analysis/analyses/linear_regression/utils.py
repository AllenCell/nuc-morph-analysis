def list_of_strings(arg):
    return arg.split(",")


def list_of_floats(arg):
    return list(map(float, arg.split(",")))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False
