#!/usr/bin/env python
# -*- coding: utf-8 -*-


def get_rng(dset):
    rng = list(range(dset["lb"], dset["ub"] + 1, dset["step"]))
    if (dset["ub"]) % (dset["step"]) != 0:
        rng.append(dset["ub"])

    return rng
