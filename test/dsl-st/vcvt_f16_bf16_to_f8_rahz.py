#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FpNarrow f16/bf16 -> f8 staged lowering ST (issue #1629).

A5 F2F has no direct f16/bf16 -> fp8 form, so the VMI vcvt stages through an
exact f32 widen and the rounding/saturate ride the f32 -> fp8 profile
(R/A/H/Z).  The probe sets mirror vcvt_f32_to_f8_rahz.py: the values are
f16-exact, and for bf16 sources the goldens hold for the bf16-quantized
payload that actually reaches the kernel (e.g. 1e10 is rounded to bf16
first).  Each set mixes exact grid points, half-way ties and one non-tie
value (1.09375 for e4m3, 1.1875 for e5m2) so Z is numerically distinguishable
from R/A.  The A5 H form empirically truncates like Z on every probed
f32 -> fp8 value (ties and non-ties alike — the direct f32 -> f8 suite shows
the same aliasing), so its goldens match Z here; the structural passthrough
of rounding="H" onto the physical vcvt is pinned by the lit tests.  The f16
e5m2 set swaps the overflowing 1e10 for the exactly representable f8e5m2 max
(57344) to stay inside the f16 range.  Two NOSAT cases verify the saturate
attribute rides the staged leg on in-range values.
"""

from pathlib import Path
import sys

import ml_dtypes
import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vcvt_f16_bf16_to_f8_rahz.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 256

# 1.09375 / 1.1875 sit 3/4 of the way to the next e4m3 / e5m2 grid point, so
# every rounding mode except Z moves up: the non-tie columns distinguish H
# from Z, which the half-way tie columns cannot.
F8E4M3_VALUES = np.array(
    [1.0625, 1.1875, -1.0625, -1.1875, 1000.0, -1000.0, np.nan, 0.0, 1.09375],
    dtype=np.float32,
)
# f16-representable variant of the e5m2 probe set: 1e10 overflows f16, use
# the exactly representable f8e5m2 maximum instead.
F8E5M2_F16_VALUES = np.array(
    [1.125, 1.375, -1.125, -1.375, 57344.0, -57344.0, np.nan, 0.0, 1.1875],
    dtype=np.float32,
)
F8E5M2_VALUES = np.array(
    [1.125, 1.375, -1.125, -1.375, 1.0e10, -1.0e10, np.nan, 0.0, 1.1875],
    dtype=np.float32,
)

F8E4M3_GOLDENS = {
    pto.VcvtRoundMode.R: np.array([0x38, 0x3A, 0xB8, 0xBA, 0x7E, 0xFE, 0x00, 0x00, 0x39], dtype=np.uint8),
    pto.VcvtRoundMode.A: np.array([0x39, 0x3A, 0xB9, 0xBA, 0x78, 0xF8, 0x00, 0x00, 0x39], dtype=np.uint8),
    pto.VcvtRoundMode.H: np.array([0x38, 0x39, 0xB8, 0xB9, 0x7E, 0xFE, 0x00, 0x00, 0x38], dtype=np.uint8),
    pto.VcvtRoundMode.Z: np.array([0x38, 0x39, 0xB8, 0xB9, 0x7E, 0xFE, 0x00, 0x00, 0x38], dtype=np.uint8),
}
F8E5M2_GOLDENS = {
    pto.VcvtRoundMode.R: np.array([0x3C, 0x3E, 0xBC, 0xBE, 0x7B, 0xFB, 0x00, 0x00, 0x3D], dtype=np.uint8),
    pto.VcvtRoundMode.A: np.array([0x3D, 0x3E, 0xBD, 0xBE, 0x7D, 0xFD, 0x00, 0x00, 0x3D], dtype=np.uint8),
    pto.VcvtRoundMode.H: np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7C, 0xFC, 0x00, 0x00, 0x3C], dtype=np.uint8),
    pto.VcvtRoundMode.Z: np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7C, 0xFC, 0x00, 0x00, 0x3C], dtype=np.uint8),
}
# 57344 is the exact f8e5m2 maximum: every rounding mode maps it to itself,
# so the overflow-column bytes differ from the saturating 1e10 probe set.
F8E5M2_F16_GOLDENS = {
    pto.VcvtRoundMode.R: np.array([0x3C, 0x3E, 0xBC, 0xBE, 0x7B, 0xFB, 0x00, 0x00, 0x3D], dtype=np.uint8),
    pto.VcvtRoundMode.A: np.array([0x3D, 0x3E, 0xBD, 0xBE, 0x7B, 0xFB, 0x00, 0x00, 0x3D], dtype=np.uint8),
    pto.VcvtRoundMode.H: np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7B, 0xFB, 0x00, 0x00, 0x3C], dtype=np.uint8),
    pto.VcvtRoundMode.Z: np.array([0x3C, 0x3D, 0xBC, 0xBD, 0x7B, 0xFB, 0x00, 0x00, 0x3C], dtype=np.uint8),
}
# In-range finite probes for the NOSAT cases: SAT and NOSAT coincide on
# every value, so these pin the saturate-attribute passthrough without
# depending on the NOSAT overflow bit-truncation semantics.
NOSAT_E4M3_VALUES = np.array(
    [1.0625, 1.09375, -1.0625, -1.09375, 1.1875, -1.1875, 0.0, 0.0],
    dtype=np.float32,
)
NOSAT_E4M3_GOLDEN = np.array(
    [0x38, 0x39, 0xB8, 0xB9, 0x3A, 0xBA, 0x00, 0x00], dtype=np.uint8
)
NOSAT_E5M2_VALUES = np.array(
    [1.125, 1.1875, -1.125, -1.1875, 1.375, -1.375, 0.0, 0.0],
    dtype=np.float32,
)
NOSAT_E5M2_GOLDEN = np.array(
    [0x3C, 0x3D, 0xBC, 0xBD, 0x3E, 0xBE, 0x00, 0x00], dtype=np.uint8
)


def _repeat_pattern(pattern: np.ndarray) -> np.ndarray:
    repeats = (ELEMS + len(pattern) - 1) // len(pattern)
    return np.tile(pattern, repeats)[:ELEMS].copy()


def _build_kernel(case):
    src_dtype = case["src_dtype"]
    dst_dtype = case["dst_dtype"]
    rounding = case["rounding"]
    saturate = case["saturate"]
    kernel_name = f"{case['name']}_kernel"

    @pto.jit(
        name=kernel_name,
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(src: pto.ptr(src_dtype, "gm"), dst: pto.ptr(pto.ui8, "gm")):
        ub_src = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(src_dtype, "ub"))
        ub_dst_u8 = pto.castptr(pto.const(2048, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
        ub_dst_f8 = pto.castptr(pto.const(2048, dtype=pto.i64), pto.ptr(dst_dtype, "ub"))

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(src, ub_src, 0, 512, nburst=(1, 512, 512))
        pto.rls_buf(pto.Pipe.MTE2, 0)
        with pto.section("vector"):
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)

        offset = pto.const(0, dtype=pto.index)
        mask = pto.vmi.create_mask(pto.const(256, dtype=pto.index), size=256)
        wide = pto.vmi.vload(ub_src, offset, size=256)
        packed = pto.vmi.vcvt(
            wide,
            dst_dtype,
            rounding=rounding,
            saturate=saturate,
        )
        pto.vmi.vstore(packed, ub_dst_f8, offset, mask)
        pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
        pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_dst_u8, dst, 256, nburst=(1, 256, 256))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _make_inputs(case):
    pattern = _repeat_pattern(case["values"])
    if case["src_key"] == "bf16":
        # torch cannot carry a bfloat16 numpy view; feed the bf16 payload as
        # raw uint16 bit patterns.  The kernel pointer only consumes the data
        # handle, so the bytes reach the vload unchanged.
        return [pattern.astype(ml_dtypes.bfloat16).view(np.uint16)]
    return [pattern.astype(np.float16)]


_SPECS = []
for src_key, src_pto in (
    ("f16", pto.f16),
    ("bf16", pto.bf16),
):
    for rnd_key in ("r", "a", "h", "z"):
        rounding = getattr(pto.VcvtRoundMode, rnd_key.upper())
        _SPECS.append(
            {
                "name": f"vmi_vcvt_{src_key}_to_f8e4m3_{rnd_key}",
                "src_dtype": src_pto,
                "src_key": src_key,
                "dst_dtype": pto.f8e4m3,
                "rounding": rounding,
                "saturate": pto.VcvtSatMode.SAT,
                "values": F8E4M3_VALUES,
                "golden": F8E4M3_GOLDENS[rounding],
            }
        )
        values = F8E5M2_F16_VALUES if src_key == "f16" else F8E5M2_VALUES
        goldens = F8E5M2_F16_GOLDENS if src_key == "f16" else F8E5M2_GOLDENS
        _SPECS.append(
            {
                "name": f"vmi_vcvt_{src_key}_to_f8e5m2_{rnd_key}",
                "src_dtype": src_pto,
                "src_key": src_key,
                "dst_dtype": pto.f8e5m2,
                "rounding": rounding,
                "saturate": pto.VcvtSatMode.SAT,
                "values": values,
                "golden": goldens[rounding],
            }
        )
_SPECS.append(
    {
        "name": "vmi_vcvt_f16_to_f8e4m3_r_nosat",
        "src_dtype": pto.f16,
        "src_key": "f16",
        "dst_dtype": pto.f8e4m3,
        "rounding": pto.VcvtRoundMode.R,
        "saturate": pto.VcvtSatMode.NOSAT,
        "values": NOSAT_E4M3_VALUES,
        "golden": NOSAT_E4M3_GOLDEN,
    }
)
_SPECS.append(
    {
        "name": "vmi_vcvt_bf16_to_f8e5m2_r_nosat",
        "src_dtype": pto.bf16,
        "src_key": "bf16",
        "dst_dtype": pto.f8e5m2,
        "rounding": pto.VcvtRoundMode.R,
        "saturate": pto.VcvtSatMode.NOSAT,
        "values": NOSAT_E5M2_VALUES,
        "golden": NOSAT_E5M2_GOLDEN,
    }
)

CASES = []
KERNELS = []
for case in _SPECS:
    kernel = _build_kernel(case)
    KERNELS.append(kernel)
    CASES.append(
        golden_output_case(
            case["name"],
            kernel,
            inputs=lambda case=case: _make_inputs(case),
            expected=_repeat_pattern(case["golden"]),
            output_shape=(ELEMS,),
            output_dtype=np.uint8,
            rtol=0.0,
            atol=0.0,
        )
    )

auto_main(globals())
