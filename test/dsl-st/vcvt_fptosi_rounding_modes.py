#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""VMI ``vcvt`` fp-to-int rounding modes (issue #1627).

The VMI conversion contract allows ``rounding`` = ``R``/``A``/``F``/``C``/``Z``
for every FpToSi/FpToUi pair (``docs/isa/vmi-isa/06-convert.md``) and the
physical ``pto.vcvt`` verifier accepts exactly that set, but PTODSL used to
normalize the attribute against ``{R, A, H, Z}`` and rejected ``F``/``C``
before ptoas ever saw the op.

The case pins the whole matrix at the DSL level:

* ``f32 -> si32`` (``requiresSat = true``) for every rounding mode;
* ``f16 -> si32`` (``requiresSat = false`` -- the contract forbids
  ``saturate``, so it must be omitted) for ``F``/``C``.

Probe values are chosen so that all five modes disagree, and the golden is
computed with the matching numpy rounding primitive.
"""

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto

ELEMS = 64
UB_SRC_OFFSET = 0
UB_DST_OFFSET = 2048

ROUNDING_MODES = ("R", "A", "F", "C", "Z")

# Every value is exactly representable in f16, so the same probe drives both the
# f32 and the f16 kernels.
PROBE = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 1.4, -1.4], dtype=np.float32)


def _repeat_probe() -> np.ndarray:
    repeats = (ELEMS + PROBE.size - 1) // PROBE.size
    return np.tile(PROBE, repeats)[:ELEMS].copy()


def _round(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "R":
        return np.rint(values)
    if mode == "A":
        return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))
    if mode == "F":
        return np.floor(values)
    if mode == "C":
        return np.ceil(values)
    if mode == "Z":
        return np.trunc(values)
    raise ValueError(f"unsupported rounding probe {mode!r}")


def _golden(mode: str) -> np.ndarray:
    rounded = _round(_repeat_probe(), mode)
    info = np.iinfo(np.int32)
    return np.clip(rounded, info.min, info.max).astype(np.int32)


def _build_f32_to_si32(rounding: str):
    @pto.jit(
        name=f"vmi_vcvt_f32_to_si32_round_{rounding.lower()}",
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(src: pto.ptr(pto.f32, "gm"), dst: pto.ptr(pto.si32, "gm")):
        ub_src = pto.castptr(pto.const(UB_SRC_OFFSET, dtype=pto.i64), pto.ptr(pto.f32, "ub"))
        ub_dst = pto.castptr(pto.const(UB_DST_OFFSET, dtype=pto.i64), pto.ptr(pto.si32, "ub"))

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(src, ub_src, 0, 256, nburst=(1, 256, 256))
        pto.rls_buf(pto.Pipe.MTE2, 0)

        with pto.section("vector"):
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
            offset = pto.const(0, dtype=pto.index)
            mask = pto.vmi.create_mask(pto.const(ELEMS, dtype=pto.index), size=ELEMS)
            wide = pto.vmi.vload(ub_src, offset, size=ELEMS)
            converted = pto.vmi.vcvt(
                wide,
                pto.si32,
                rounding=rounding,
                saturate=pto.VcvtSatMode.SAT,
            )
            pto.vmi.vstore(converted, ub_dst, offset, mask)
            pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
            pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_dst, dst, 256, nburst=(1, 256, 256))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


def _build_f16_to_si32(rounding: str):
    @pto.jit(
        name=f"vmi_vcvt_f16_to_si32_round_{rounding.lower()}",
        target="a5",
        backend="vpto",
        mode="explicit",
        insert_sync=False,
    )
    def kernel(src: pto.ptr(pto.f16, "gm"), dst: pto.ptr(pto.si32, "gm")):
        ub_src = pto.castptr(pto.const(UB_SRC_OFFSET, dtype=pto.i64), pto.ptr(pto.f16, "ub"))
        ub_dst = pto.castptr(pto.const(UB_DST_OFFSET, dtype=pto.i64), pto.ptr(pto.si32, "ub"))

        pto.get_buf(pto.Pipe.MTE2, 0)
        pto.mte_gm_ub(src, ub_src, 0, 128, nburst=(1, 128, 128))
        pto.rls_buf(pto.Pipe.MTE2, 0)

        with pto.section("vector"):
            pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
            pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
            offset = pto.const(0, dtype=pto.index)
            mask = pto.vmi.create_mask(pto.const(ELEMS, dtype=pto.index), size=ELEMS)
            narrow = pto.vmi.vload(ub_src, offset, size=ELEMS)
            # f16 -> si32 is a widening pair whose contract sets
            # requiresSat=false: passing "saturate" here would be rejected.
            converted = pto.vmi.vcvt(narrow, pto.si32, rounding=rounding)
            pto.vmi.vstore(converted, ub_dst, offset, mask)
            pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
            pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)

        pto.get_buf(pto.Pipe.MTE3, 0)
        pto.mte_ub_gm(ub_dst, dst, 256, nburst=(1, 256, 256))
        pto.rls_buf(pto.Pipe.MTE3, 0)
        pto.pipe_barrier(pto.Pipe.ALL)

    return kernel


KERNELS = []
CASES = []

for mode in ROUNDING_MODES:
    f32_kernel = _build_f32_to_si32(mode)
    KERNELS.append(f32_kernel)
    CASES.append(
        golden_output_case(
            f"vmi_vcvt_f32_to_si32_round_{mode.lower()}",
            f32_kernel,
            inputs=lambda: [_repeat_probe().astype(np.float32)],
            expected=_golden(mode),
            output_shape=(ELEMS,),
            output_dtype=np.int32,
            rtol=0.0,
            atol=0.0,
        )
    )

for mode in ("F", "C"):
    f16_kernel = _build_f16_to_si32(mode)
    KERNELS.append(f16_kernel)
    CASES.append(
        golden_output_case(
            f"vmi_vcvt_f16_to_si32_round_{mode.lower()}",
            f16_kernel,
            inputs=lambda: [_repeat_probe().astype(np.float16)],
            expected=_golden(mode),
            output_shape=(ELEMS,),
            output_dtype=np.int32,
            rtol=0.0,
            atol=0.0,
        )
    )

auto_main(globals())
