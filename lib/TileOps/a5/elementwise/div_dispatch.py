# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Element type table and division dispatch shared by pto.tdiv and pto.tdivs.

Both templates accept the same signatures and pick between the same
implementations, so the table and the dispatch live here instead of being
duplicated in each template module.

The dispatch mirrors AscendC's ``DivImpl`` for the build PTO targets (FTZ on):

* f32: ``intrinsic`` / ``precision_1ulp_ftz_true`` are a native divide;
  ``diff_compensation`` and ``precision_0ulp_ftz_true`` use the compensated
  residual search; ``precision_0ulp_ftz_false`` / ``precision_1ulp_ftz_false``
  use the IEEE-754 normalized path.
* f16: only ``precision_1ulp_ftz_false`` has an implementation
  (``DivIEEE754HalfImpl``); the 0-ULP algorithms are f32-only and are rejected
  here just like the reference rejects them with a static_assert.
* Every other element type falls back to the native vector divide.

The f32 software algorithms are imported from ``SoftOps`` so the tile path and
the VMI path share one implementation and cannot drift apart.
"""

from SoftOps import div_i32_soft
from SoftOps.div_vector import (
    div_vf32_soft,
    div_vf32_ulp0_ftz_false,
    div_vf32_ulp1_ftz_false,
)

from ptodsl import pto

from .div_hp import _div_ieee754_f16_impl


DIV_DTYPES = [(dtype,) * 3 for dtype in ("f16", "f32", "i32", "i16", "ui16", "ui32")]

# f32 selectors served by the Software Library.
F32_SOFTWARE = {
    # DIFF_COMPENSATION maps to the compensated DivPrecisionImpl in the FTZ build
    # of the reference, i.e. the same sequence as the 0-ULP FTZ-true algorithm.
    "diff_compensation": div_vf32_soft,
    "precision_0ulp_ftz_true": div_vf32_soft,
    "precision_0ulp_ftz_false": div_vf32_ulp0_ftz_false,
    "precision_1ulp_ftz_false": div_vf32_ulp1_ftz_false,
}

# f16 selectors served by the Software Library (DivIEEE754HalfImpl).
F16_SOFTWARE = {
    "precision_1ulp_ftz_false": _div_ieee754_f16_impl,
}

# 0-ULP algorithms have no f16 implementation in the reference.
F16_UNSUPPORTED = ("precision_0ulp_ftz_true", "precision_0ulp_ftz_false")


def divide(lhs, rhs, dtype, mask, precision_type):
    """Select the division implementation for one element type.

    ``precisionType`` names an AscendC ``DivAlgo`` algorithm. Integer division is
    already exact, so a precision selector is accepted and ignored on integer
    tiles rather than rejected; the dtype guard is also what keeps integers out
    of the float paths.
    """
    name = str(dtype)
    if name == "i32":
        # i32 stays inlined in the template rather than deferring to the late
        # SoftLib expansion of pto.vdiv, which would also accept it. The
        # expanded form is pinned by test/lit/vpto/issue_1181_tdiv_i32_tilelib_
        # expand.pto, which requires the signed helper's ops to appear directly
        # in the tile-op-expand output with no integer pto.vdiv left behind.
        return div_i32_soft(lhs, rhs, mask)
    if name == "f32" and precision_type in F32_SOFTWARE:
        return F32_SOFTWARE[precision_type](lhs, rhs, mask)
    if name == "f16":
        if precision_type in F16_SOFTWARE:
            return F16_SOFTWARE[precision_type](lhs, rhs, mask)
        if precision_type in F16_UNSUPPORTED:
            raise ValueError(
                f"{precision_type} is f32-only; f16 supports "
                "precision_1ulp_ftz_false or a native selector"
            )
    return pto.vdiv(lhs, rhs, mask)
