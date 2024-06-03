import numpy as np

_SI_PREFIXES = ["", "K", "M", "G", "T", "P", "E"]


def to_si(number, precision=1):
    if number == 0:
        return "{short_number:.{precision}f}".format(short_number=0., precision=precision)
    if number < 0:
        number = -number
        sign = "-"
    else:
        sign = ""

    magnitude = int(np.log10(number) / 3)
    short_number = int(number / (1000 ** magnitude / 10 ** precision)) / 10 ** precision
    return "{sign}{short_number:.{precision}f}{si_unit}".format(
        sign=sign,
        short_number=short_number,
        precision=precision,
        si_unit=_SI_PREFIXES[magnitude],
    )


def dict_to_string(obj, seperator="_"):
    """ {epoch: 5, batchsize: 64} --> epoch=5_batchsize=64 """
    assert isinstance(obj, dict)
    for key in obj.keys():
        if seperator in key:
            raise NotImplementedError(f"using '{seperator}' in a variable name is not supported ('{key}')")
    return seperator.join(f"{k}={v}" for k, v in obj.items())
