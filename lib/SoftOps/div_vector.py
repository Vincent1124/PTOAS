# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Register-level residual correction: the 21-instruction 0-ULP divide.

This mirrors the *unscaled* ``DivPrecisionImpl`` -- the form the AscendC SIMD
dialect emits as ``vdiv_0ulp_ftz_true`` (``tilelang`` ``simd_inst.h``) and the
form of ``dav_m510``'s reference -- operation for operation:

    z      = vdiv(lhs, rhs)
    y      = -rhs
    r      = |lhs*z + y|                        (fused, one rounding)
    zPre   = bitwise neighbour of z (-1)
    zNext  = bitwise neighbour of z (+1)
    rPre   = |lhs*zPre + y|, rNext = |lhs*zNext + y|
    the candidate with the smallest residual wins; an equal residual keeps the
    lower neighbour (z - 1), matching the reference ``vcmp_lt`` + ``vsel(z, z,
    zPre)`` chain -- there is no round-half-even tie breaking
    special lanes (inf / nan / zero) keep the native value

``dav_3510``'s ``DivPrecisionImpl`` additionally pre-scales both operands by
``2^k`` when the unbiased exponent of ``lhs`` is below ``-64``. That block is
deliberately **not** implemented here: it costs 11 extra vector instructions per
division, and the unscaled sequence matches the AscendC template the ASC backend
compiles today. Consequence under flush-to-zero: for ``lhs`` exponents below
about ``-102`` the residual ``|a - q*b| ~ 2^(e0-24)`` can be flushed to zero, the
tie rule then keeps ``z - 1``, and the result can be one ULP away from the native
divide. Use ``precision_0ulp_ftz_false`` when that corner has to be exact.
"""

from ptodsl import pto


def div_vf32_soft(lhs, rhs, mask):
    """Preserve native special results and apply the CANN 0-ULP f32 correction."""
    quotient = pto.vdiv(lhs, rhs, mask)
    bits = pto.vbitcast(quotient, pto.ui32)
    tagged = pto.vor(bits, pto.vbr(pto.ui32(0x80000000)), mask)
    special = pto.vcmps(tagged, pto.ui32(0xFF800000), mask, pto.CmpMode.GE)
    zero = pto.vcmps(quotient, pto.f32(0.0), mask, pto.CmpMode.EQ)
    special = pto.por(special, zero, mask)
    negative_rhs = pto.vmuls(rhs, pto.f32(-1.0), mask)
    # Residual |lhs - q*rhs| (CANN's vmula(r=src0, z, y) form), kept fused so the
    # comparison is not perturbed by an intermediate rounding.
    residual = pto.vabs(pto.vmula(lhs, quotient, negative_rhs, mask), mask)
    previous = pto.vbitcast(
        pto.vadds(pto.vbitcast(quotient, pto.si32), pto.si32(-1), mask), pto.f32
    )
    following = pto.vbitcast(
        pto.vadds(pto.vbitcast(quotient, pto.si32), pto.si32(1), mask), pto.f32
    )
    residual_previous = pto.vabs(pto.vmula(lhs, previous, negative_rhs, mask), mask)
    residual_following = pto.vabs(pto.vmula(lhs, following, negative_rhs, mask), mask)
    # current_better is "the native quotient already beats its lower neighbour".
    current_better = pto.vcmp(residual, residual_previous, mask, pto.CmpMode.LT)
    best = pto.vsel(quotient, previous, current_better)
    residual = pto.vsel(residual, residual_previous, current_better)
    following_better = pto.vcmp(residual_following, residual, mask, pto.CmpMode.LT)
    best = pto.vsel(following, best, following_better)
    return pto.vsel(quotient, best, special)


def div_vf32_ulp0_ftz_false(lhs, rhs, mask):
    """AscendC ``DivAlgo::PRECISION_0ULP_FTZ_FALSE`` for A5 f32.

    Normalises subnormal operands, resolves the quotient with the residual
    search in the normalised domain, then re-applies the exponent and performs
    IEEE-754 subnormal/overflow rounding (``DivIEEE754FloatImpl<is0ULP=true>``).
    """
    from SoftOps.div_ieee754 import div_vf32_ieee754

    return div_vf32_ieee754(lhs, rhs, mask, ulp0=True)


def div_vf32_ulp1_ftz_false(lhs, rhs, mask):
    """AscendC ``DivAlgo::PRECISION_1ULP_FTZ_FALSE`` for A5 f32.

    Same normalisation and rounding path as the 0-ULP variant but the normalised
    quotient comes from a single hardware divide
    (``DivIEEE754FloatImpl<is0ULP=false>``).
    """
    from SoftOps.div_ieee754 import div_vf32_ieee754

    return div_vf32_ieee754(lhs, rhs, mask, ulp0=False)
