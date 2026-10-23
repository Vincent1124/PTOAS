# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""IEEE-754 f32 division algorithms for AscendC ``DivAlgo`` parity.

Transcribed from the A5 reference in the AscendC devkit,
``impl/basic_api/reg_compute/dav_3510/kernel_reg_compute_vec_binary_impl.h``:

* ``DivIEEE754FloatImpl<is0ULP=false>`` -> ``DivAlgo::PRECISION_1ULP_FTZ_FALSE``
* ``DivIEEE754FloatImpl<is0ULP=true>``  -> ``DivAlgo::PRECISION_0ULP_FTZ_FALSE``
  (its normalised quotient comes from ``DivPrecisionImpl``)

with the helpers ``DivF32StickyDir`` and ``DivF32ScaleRound``. The transcription
keeps the reference order of operations so the emitted vector sequence matches
the AscendC kernel instruction for instruction.
"""

from ptodsl import pto

_F32_INF = 0x7F800000
_F32_NAN = 0x7FC00000
_SIGN_BIT = 0x80000000
_EXP_MASK = 0x7F800000
_MANTISSA_MASK = 0x007FFFFF
_IMPLICIT_BIT = 0x00800000
_EXP_NORMALIZER = 0x3F800000
_EXPONENT_EXTRACTOR = 0x807FFFFF
_NORMALIZE_ENLARGE = 0x4B000000  # 2^23


def _u32(value):
    return pto.vbr(pto.ui32(value))


def _i32(value):
    return pto.vbr(pto.si32(value))


def _f32(value):
    return pto.vbr(pto.f32(value))


def _classify_operands(src0, src1, mask):
    """Absolute values, inf/zero predicates and the subnormal masks."""
    src0_abs = pto.vabs(src0, mask)
    src1_abs = pto.vabs(src1, mask)

    # The reference bit-puns the absolutes to uint32 for the inf/zero tests.
    src0_bits = pto.vbitcast(src0_abs, pto.ui32)
    src1_bits = pto.vbitcast(src1_abs, pto.ui32)
    inf = _u32(_F32_INF)
    invalid = pto.por(
        pto.vcmp(src0_bits, inf, mask, pto.CmpMode.EQ),
        pto.vcmp(src1_bits, inf, mask, pto.CmpMode.EQ),
        mask,
    )

    zero = _u32(0)
    invalid = pto.por(invalid, pto.vcmp(src0_bits, zero, mask, pto.CmpMode.EQ), mask)
    invalid = pto.por(invalid, pto.vcmp(src1_bits, zero, mask, pto.CmpMode.EQ), mask)
    valid = pto.pnot(invalid, mask)

    min_normal = _f32(float.fromhex('0x1p-126'))
    src0_subnormal = pto.vcmp(src0_abs, min_normal, mask, pto.CmpMode.LT)
    src0_normal = pto.pnot(src0_subnormal, mask)
    src1_subnormal = pto.vcmp(src1_abs, min_normal, mask, pto.CmpMode.LT)
    src1_normal = pto.pnot(src1_subnormal, mask)

    return {
        'src0_abs': src0_abs,
        'src1_abs': src1_abs,
        'src0_subnormal': src0_subnormal,
        'src0_normal': src0_normal,
        'src1_subnormal': src1_subnormal,
        'src1_normal': src1_normal,
        'valid': valid,
    }


def _normalize_operands(src0, src1, mask, state):
    """Scale subnormal operands by 2^23 and standardise both exponents."""
    enlarge = pto.f32(float.fromhex('0x1p23'))
    src0_sub_scaled = pto.vmuls(src0, enlarge, state['src0_subnormal'])
    src1_sub_scaled = pto.vmuls(src1, enlarge, state['src1_subnormal'])

    src0_all = pto.vsel(src0, src0_sub_scaled, state['src0_normal'])
    src1_all = pto.vsel(src1, src1_sub_scaled, state['src1_normal'])

    valid = state['valid']
    extractor = _u32(_EXPONENT_EXTRACTOR)
    normalizer = _u32(_EXP_NORMALIZER)
    src0_norm = pto.vadd(
        pto.vand(pto.vbitcast(src0_all, pto.ui32), extractor, valid), normalizer, valid
    )
    src1_norm = pto.vadd(
        pto.vand(pto.vbitcast(src1_all, pto.ui32), extractor, valid), normalizer, valid
    )
    src0_norm = pto.vsel(pto.vbitcast(src0_norm, pto.f32), src0_all, valid)
    src1_norm = pto.vsel(pto.vbitcast(src1_norm, pto.f32), src1_all, valid)

    return {
        'src0_all': src0_all,
        'src1_all': src1_all,
        'src0_norm': src0_norm,
        'src1_norm': src1_norm,
    }


def _normalize(src0, src1, mask):
    """Acquire the normalised operands and the masks shared by both variants."""
    state = _classify_operands(src0, src1, mask)
    state.update(_normalize_operands(src0, src1, mask, state))
    return state


def _sticky_direction(state, quotient, mask):
    """``DivF32StickyDir``: sticky-positive / sticky-zero masks of the residual."""
    src0_norm = state['src0_norm']
    neg_src1 = pto.vmuls(state['src1_norm'], pto.f32(-1.0), mask)
    residual = pto.vmula(src0_norm, quotient, neg_src1, mask)

    gt_zero = pto.vcmps(residual, pto.f32(0.0), mask, pto.CmpMode.GT)
    lt_zero = pto.vcmps(residual, pto.f32(0.0), mask, pto.CmpMode.LT)
    sticky_zero = pto.vcmps(residual, pto.f32(0.0), mask, pto.CmpMode.EQ)

    src0_sign = pto.vand(pto.vbitcast(src0_norm, pto.ui32), _u32(_SIGN_BIT), mask)
    src0_positive = pto.vcmp(src0_sign, _u32(0), mask, pto.CmpMode.EQ)
    src0_negative = pto.pnot(src0_positive, mask)

    same_sign = pto.pand(gt_zero, src0_positive, mask)
    other_sign = pto.pand(lt_zero, src0_negative, mask)
    return pto.por(same_sign, other_sign, mask), sticky_zero


def _scale_round(dst, scale, sticky, mask):
    """``DivF32ScaleRound``: exponent re-application and subnormal rounding."""
    sign_mask = _u32(_SIGN_BIT)
    mantissa_mask = _u32(_MANTISSA_MASK)

    dst_bits = pto.vbitcast(dst, pto.ui32)
    q_sign = pto.vand(dst_bits, sign_mask, mask)
    q_exp = pto.vshrs(pto.vand(dst_bits, _u32(_EXP_MASK), mask), pto.i16(23), mask)
    e = pto.vadd(pto.vsub(pto.vbitcast(q_exp, pto.si32), _i32(127), mask), scale, mask)

    q_mantissa = pto.vand(dst_bits, mantissa_mask, mask)
    m24 = pto.vor(q_mantissa, _u32(_IMPLICIT_BIT), mask)

    overflow = pto.vcmp(e, _i32(127), mask, pto.CmpMode.GT)
    inf_bits = pto.vor(q_sign, _u32(_F32_INF), overflow)
    dst = pto.vsel(pto.vbitcast(inf_bits, pto.f32), dst, overflow)

    not_overflow = pto.pnot(overflow, mask)
    normal = pto.vcmp(e, _i32(-126), not_overflow, pto.CmpMode.GE)
    normal_exp = pto.vshls(pto.vadds(e, pto.si32(127), normal), pto.i16(23), normal)
    normal_bits = pto.vor(q_sign, pto.vbitcast(normal_exp, pto.ui32), normal)
    normal_bits = pto.vor(normal_bits, q_mantissa, normal)
    dst = pto.vsel(pto.vbitcast(normal_bits, pto.f32), dst, normal)

    subnormal = pto.pnot(normal, not_overflow)
    drop = pto.vsub(_i32(-126), e, subnormal)
    zero_mask = pto.vcmp(drop, _i32(24), subnormal, pto.CmpMode.GT)
    dst = pto.vsel(pto.vbitcast(q_sign, pto.f32), dst, zero_mask)

    round_mask = pto.pnot(zero_mask, subnormal)
    one = _u32(1)
    drop_u = pto.vbitcast(drop, pto.ui32)
    kept = pto.vshr(m24, drop_u, round_mask)
    rem_mask = pto.vsub(pto.vshl(one, drop_u, round_mask), one, round_mask)
    rem = pto.vand(m24, rem_mask, round_mask)
    drop_minus_one = pto.vsub(drop, _i32(1), round_mask)
    half = pto.vshl(one, pto.vbitcast(drop_minus_one, pto.ui32), round_mask)

    rem_gt_half = pto.vcmp(rem, half, round_mask, pto.CmpMode.GT)
    rem_eq_half = pto.vcmp(rem, half, round_mask, pto.CmpMode.EQ)
    kept_lsb = pto.vand(kept, one, round_mask)
    kept_odd = pto.vcmp(kept_lsb, _u32(0), round_mask, pto.CmpMode.NE)

    if sticky is None:
        tie_up = pto.pand(rem_eq_half, kept_odd, round_mask)
    else:
        sticky_positive, sticky_zero = sticky
        tie_up = pto.por(
            sticky_positive, pto.pand(sticky_zero, kept_odd, round_mask), round_mask
        )
        tie_up = pto.pand(tie_up, rem_eq_half, round_mask)
    round_up = pto.por(rem_gt_half, tie_up, round_mask)
    kept = pto.vsel(pto.vadd(kept, one, round_up), kept, round_up)

    result_bits = pto.vor(q_sign, kept, round_mask)
    return pto.vsel(pto.vbitcast(result_bits, pto.f32), dst, round_mask)


def _quotient_scale(state, mask):
    """Exponent difference ``e0 - e1``, corrected for scaled subnormals."""
    exp_mask = _u32(_EXP_MASK)
    src0_exp = pto.vshrs(pto.vand(pto.vbitcast(state['src0_all'], pto.ui32), exp_mask, mask),
                         pto.i16(23), mask)
    src1_exp = pto.vshrs(pto.vand(pto.vbitcast(state['src1_all'], pto.ui32), exp_mask, mask),
                         pto.i16(23), mask)
    scale = pto.vsub(pto.vbitcast(src0_exp, pto.si32), pto.vbitcast(src1_exp, pto.si32), mask)
    # AscendC keeps the out-of-mask lanes with an explicit select: a masked
    # scalar add zeroes the inactive lanes.
    src0_sub = state['src0_subnormal']
    scale = pto.vsel(pto.vadds(scale, pto.si32(-23), src0_sub), scale, src0_sub)
    src1_sub = state['src1_subnormal']
    return pto.vsel(pto.vadds(scale, pto.si32(23), src1_sub), scale, src1_sub)


def _apply_nan(state, dst, mask):
    """Propagate NaN operands into the result."""
    nan = pto.por(
        pto.vcmp(state['src0_abs'], state['src0_abs'], mask, pto.CmpMode.NE),
        pto.vcmp(state['src1_abs'], state['src1_abs'], mask, pto.CmpMode.NE),
        mask,
    )
    return pto.vsel(pto.vbitcast(_u32(_F32_NAN), pto.f32), dst, nan)


def div_vf32_ieee754(lhs, rhs, mask, *, ulp0):
    """``DivIEEE754FloatImpl`` for A5 f32."""
    state = _normalize(lhs, rhs, mask)
    if ulp0:
        from SoftOps.div_vector import div_vf32_soft

        dst = div_vf32_soft(state['src0_norm'], state['src1_norm'], mask)
        sticky = _sticky_direction(state, dst, state['valid'])
    else:
        dst = pto.vdiv(state['src0_norm'], state['src1_norm'], mask)
        sticky = None
    dst = _scale_round(dst, _quotient_scale(state, mask), sticky, state['valid'])
    return _apply_nan(state, dst, mask)
