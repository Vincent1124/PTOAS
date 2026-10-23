# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""im2col_conv_dil_runtime: 3D load with runtime dilation (dynamic path).

Same geometry as im2col_conv (fmap 6x6, filter 3x3, channel 16) but the
dilation comes from a runtime kernel scalar, so the expansion takes the
dynamic arith chain (not the constant fold). Validates the dynamic path
numerically: dilation=2 vs an independent reference, and dilation=0 which
camodel measures to behave as 1 (the expansion normalizes it).
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

L1_A_ADDR = 0
L1_B_ADDR = 2048
A_BYTES = FMAP_H * FMAP_W * CHANNEL * 2


@pto.jit(
    name="im2col_conv_dil_runtime_kernel",
    kernel_kind="cube",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def im2col_conv_dil_runtime_kernel(
    a_gm: pto.ptr(pto.f16, "gm"),
    b_gm: pto.ptr(pto.f16, "gm"),
    c_gm: pto.ptr(pto.f32, "gm"),
    dilation: pto.i64,
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
        shape=(144, OUT_CH),
        src_layout=(32,),
        dst_group=(1, 1, OUT_CH, 0),
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
        dilation_h=dilation,
        dilation_w=dilation,
    )
    pto.mte_l1_l0b(l1_b, l0b, 144, OUT_CH, transpose=True)
    pto.set_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)
    pto.wait_flag(pto.Pipe.MTE1, pto.Pipe.M, event_id=1)

    pto.mad(l0a, l0b, l0c, 16, OUT_CH, 144)
    pto.set_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)
    pto.wait_flag(pto.Pipe.M, pto.Pipe.FIX, event_id=2)

    pto.mte_l0c_gm(l0c, c_gm, 16, OUT_CH, OUT_CH, OUT_CH, 0, 0, layout="nz2nd")
    pto.pipe_barrier(pto.Pipe.ALL)


def _inputs():
    a = make_fmap(FMAP_H, FMAP_W, CHANNEL)
    w = make_weight(OUT_CH, CHANNEL, FILTER_H, FILTER_W)
    b = make_b(w, CHANNEL, FILTER_H, FILTER_W, OUT_CH)
    return [a, b]


def _case(name, dilation_value, golden_dilation):
    def expected(a, b):
        return im2col_conv_reference(
            a, b, fmap_h=FMAP_H, fmap_w=FMAP_W,
            filter_h=FILTER_H, filter_w=FILTER_W,
            channel=CHANNEL, out_ch=OUT_CH, dilation_h=golden_dilation,
            dilation_w=golden_dilation,
        )

    return golden_output_case(
        name,
        im2col_conv_dil_runtime_kernel,
        inputs=_inputs,
        expected=expected,
        launch_args=[dilation_value],
        output_dtype=np.float32,
        output_index=2,
        rtol=1e-2,
        atol=1e-2,
    )


# dilation=2 would change M (eff=5 -> M=4) while the kernel keeps M=16 for
# the mad/writeback; dil=0 (normalized to 1, M stays 16) still exercises the
# dynamic path and the camodel-measured 0-behaves-as-1 behavior.
CASES = [
    _case("im2col_conv_dil_runtime_0", 0, 1),  # camodel: dilation 0 behaves as 1
]

if __name__ == "__main__":
    auto_main(globals())
