# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""im2col_conv: A5 LOAD L1 TO L0A 3Dv2 im2col + MMAD end-to-end (DSL ST).

fmap = 6x6, C0 = 16 (f16), filter = 3x3, stride = 1, no pad/dilation.
H_o = W_o = 4 -> M = 16, K = 144, N = 16.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

from _im2col_reference import im2col_conv_reference, make_b, make_fmap, make_weight

FMAP_H = 6
FMAP_W = 6
FILTER_H = 3
FILTER_W = 3
CHANNEL = 16
OUT_CH = 16
M = (FMAP_H - FILTER_H + 1) * (FMAP_W - FILTER_W + 1)
K = CHANNEL * FILTER_H * FILTER_W
N = OUT_CH

L1_A_ADDR = 0
L1_B_ADDR = 2048
A_BYTES = FMAP_H * FMAP_W * CHANNEL * 2


@pto.jit(
    name="im2col_conv_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def im2col_conv_kernel(
    a_gm: pto.ptr(pto.f16, "gm"),
    b_gm: pto.ptr(pto.f16, "gm"),
    c_gm: pto.ptr(pto.f32, "gm"),
):
    l1_a = pto.castptr(pto.ui64(L1_A_ADDR), pto.ptr(pto.f16, "mat"))
    l1_b = pto.castptr(pto.ui64(L1_B_ADDR), pto.ptr(pto.f16, "mat"))
    l0a = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "left"))
    l0b = pto.castptr(pto.ui64(0), pto.ptr(pto.f16, "right"))
    l0c = pto.castptr(pto.ui64(0), pto.ptr(pto.f32, "acc"))

    pto.mte_gm_l1(a_gm, l1_a, A_BYTES, nburst=(1, 0, 0))
    pto.mte_gm_l1_frac(
        b_gm,
        l1_b,
        pto.FractalMode.ND2NZ,
        shape=(K, N),
        src_layout=(32,),
        dst_group=(1, 1, N, 0),
        ctrl=(0, False),
    )
    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.MTE1, event_id=0)

    pto.mte_l1_l0a_3d(
        l1_a,
        l0a,
        fmap_h=FMAP_H,
        fmap_w=FMAP_W,
        filter_h=FILTER_H,
        filter_w=FILTER_W,
        channel_size=CHANNEL,
    )
    pto.mte_l1_l0b(l1_b, l0b, K, N, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)

    pto.mad(l0a, l0b, l0c, M, N, K)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)

    pto.mte_l0c_gm(l0c, c_gm, M, N, N, N, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _inputs():
    a = make_fmap(FMAP_H, FMAP_W, CHANNEL)
    w = make_weight(OUT_CH, CHANNEL, FILTER_H, FILTER_W)
    b = make_b(w, CHANNEL, FILTER_H, FILTER_W, OUT_CH)
    return [a, b]


def _expected(a, b):
    return im2col_conv_reference(
        a, b, fmap_h=FMAP_H, fmap_w=FMAP_W,
        filter_h=FILTER_H, filter_w=FILTER_W,
        channel=CHANNEL, out_ch=OUT_CH,
    )


CASES = [
    golden_output_case(
        "im2col_conv",
        im2col_conv_kernel,
        inputs=_inputs,
        expected=_expected,
        output_shape=(M, N),
        output_dtype=np.float32,
        output_index=2,
        rtol=1e-2,
        atol=1e-2,
    ),
]

if __name__ == "__main__":
    auto_main(globals())
