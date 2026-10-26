#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""SIMT shuffle source window (ISA SHFL Rm[12:8] / lane_mask) numeric check.

Each lane shuffles its own lane id and stores the result. The reference follows
the ISA semantics:

    group_base = (lane / width) * width
    max_lane   = group_base + bound % width
    in_window  = (mode == up) ? source >= max_lane : source <= max_lane
    result     = value[source] when in_window, else value[lane]

Cases cover an explicit upper bound (shuffle_down), an explicit lower bound
(shuffle_up) and the canonical full window of a narrow (width = 8) subgroup.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


LANES = 32
WIDTH = 8
DOWN_OFFSET = 1
DOWN_BOUND = 3
UP_OFFSET = 2
UP_BOUND = 5


@pto.simt
def simt_shuffle_source_window_body(out: pto.ptr(pto.i32, "gm")):
    lane = pto.get_laneid()
    narrowed_down = pto.shuffle_down(
        lane, DOWN_OFFSET, width=WIDTH, bound=DOWN_BOUND
    )
    narrowed_up = pto.shuffle_up(lane, UP_OFFSET, width=WIDTH, bound=UP_BOUND)
    full_window_down = pto.shuffle_down(lane, DOWN_OFFSET, width=WIDTH)
    pto.store(narrowed_down, out, lane)
    pto.store(narrowed_up, out, lane + LANES)
    pto.store(full_window_down, out, lane + 2 * LANES)


@pto.jit(
    name="simt_shuffle_source_window_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def simt_shuffle_source_window_kernel(out_ptr: pto.ptr(pto.i32, "gm")):
    simt_shuffle_source_window_body[LANES, 1, 1](out_ptr)
    pto.pipe_barrier(pto.Pipe.ALL)


def _shuffle(mode, lane, argument, width, bound=None):
    group_base = (lane // width) * width
    local_lane = lane - group_base
    if bound is None:
        bound = 0 if mode == "up" else width - 1
    max_lane = group_base + (bound % width)
    if mode == "down":
        source = group_base + local_lane + argument
    elif mode == "up":
        source = group_base + local_lane - argument
    else:
        source = group_base + (local_lane ^ argument)
    in_window = source >= max_lane if mode == "up" else source <= max_lane
    return np.where(in_window, source, lane)


def make_inputs():
    return []


def make_expected():
    lane = np.arange(LANES, dtype=np.int32)
    narrowed_down = _shuffle("down", lane, DOWN_OFFSET, WIDTH, DOWN_BOUND)
    narrowed_up = _shuffle("up", lane, UP_OFFSET, WIDTH, UP_BOUND)
    full_window_down = _shuffle("down", lane, DOWN_OFFSET, WIDTH)
    return np.concatenate([narrowed_down, narrowed_up, full_window_down]).astype(
        np.int32
    )


CASES = [
    golden_output_case(
        "simt_shuffle_source_window",
        simt_shuffle_source_window_kernel,
        inputs=make_inputs,
        expected=make_expected(),
        output_shape=(3 * LANES,),
        output_dtype=np.int32,
        output_index=0,
        rtol=0,
        atol=0,
    ),
]


if __name__ == "__main__":
    auto_main(globals())
