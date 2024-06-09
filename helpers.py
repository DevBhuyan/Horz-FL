#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_rng(dset):
    rng = list(range(dset["lb"], dset["ub"] + 1, dset["step"]))
    if (dset["ub"]) % (dset["step"]) != 0:
        rng.append(dset["ub"])

    return rng


def flashing_progress(current: int,
                      total: int,
                      desc: str = "Progress"):
    ratio = current / (total-1)
    bar_length = 20
    bar_filled = int(ratio * bar_length)
    bar_empty = bar_length - bar_filled
    bar = "[" + "█" * bar_filled + "-" * bar_empty + "]"

    print("\r{}: {} {:.2%}".format(desc, bar, ratio), end="")
