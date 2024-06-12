#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_rng(dset):
    """
    Generate a range of numbers from the dataset parameters.

    Parameters
    ----------
    dset : dict
        A dictionary containing the keys 'lb' (lower bound), 'ub' (upper bound), and 'step'.

    Returns
    -------
    rng : list
        A list of integers from 'lb' to 'ub' with steps of 'step'.
    """
    rng = list(range(dset["lb"], dset["ub"] + 1, dset["step"]))
    if (dset["ub"]) % (dset["step"]) != 0:
        rng.append(dset["ub"])

    return rng


def flashing_progress(current: int,
                      total: int,
                      desc: str = "Progress"):
    """
    Display a progress bar in the console.

    Parameters
    ----------
    current : int
        The current progress count.
    total : int
        The total count for completion.
    desc : str, optional
        A description to display alongside the progress bar. Default is "Progress".
    """
    ratio = current / (total-1)
    bar_length = 20
    bar_filled = int(ratio * bar_length)
    bar_empty = bar_length - bar_filled
    bar = "[" + "█" * bar_filled + "-" * bar_empty + "]"

    print("\r{}: {} {:.2%}".format(desc, bar, ratio), end="")
