# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""IEEE-754 float16 division for the A5 tile path.

``tdiv`` / ``tdivs`` route ``precision_1ulp_ftz_false`` on f16 here; every other
f16 selector lowers to the native vector divide, and the 0-ULP algorithms have no
f16 implementation (AscendC rejects them with a static_assert). The f32 software
algorithms share the SoftOps implementation in ``SoftOps.div_ieee754`` so the
tile path and the VMI path cannot drift apart.

``_vmula`` is intentionally gone: the reference residuals are fused, and the f32
three-candidate search now lives in the SoftOps module.
"""

from dataclasses import dataclass

from ptodsl import pto


@dataclass


class _F16NormalizedState:
    """Normalized f16 division operands and masks shared by the fix-up phases."""

    src0_norm: object
    src1_norm: object
    src0_abs: object
    src1_abs: object
    src0_all_u16: object
    src1_all_u16: object
    mask_src0_subnormal: object
    mask_src0_normal: object
    mask_src1_subnormal: object
    mask_src1_normal: object
    mask_valid: object


def _div_f16_normalize(src0, src1, mask):
    """Classify f16 specials/subnormals and re-bias the operand exponents.

    Symmetric subnormal handling (both operands use LT) matching
    Div754.hpp:291-401. Returns the normalized operands plus the masks
    consumed by the fix-up phases.
    """
    f16_inf = pto.ui16(0x7C00)              # +Infinity: sign=0, exp=31, mant=0
    exponent_extractor = pto.ui16(0x83FF)   # Clear exponent bits [14:10]
    exponent_normalizer = pto.ui16(0x3C00)  # 1.0f16 reference (bias=15)
    sign_extractor = pto.ui16(0x8000)       # Sign bit mask (bit15)
    subnormal_threshold = pto.ui16(0x0400)  # Min normal: 2^-14 (dav_3510 uses LT here)
    nan_value = pto.ui16(0x7E00)            # Quiet NaN: exp=31, mant=0x200
    min_denormal = pto.ui16(0x1)            # Smallest positive: 2^-24

    # Subnormal normalization factors (corresponds to Div754.hpp:306-309)
    normalize_scale_enlarge = pto.f16(1024.0)             # 2^10: shifts subnormals to normal range
    normalize_scale_reduce = pto.f16(0.0009765625)        # 2^-10: inverse for result compensation

    src0_abs = pto.vabs(src0, mask)
    src1_abs = pto.vabs(src1, mask)

    src0_abs_u16 = pto.vbitcast(src0_abs, pto.ui16)
    src1_abs_u16 = pto.vbitcast(src1_abs, pto.ui16)

    # Detect Infinity values
    mask_inf_src0 = pto.vcmp(src0_abs_u16, pto.vbr(f16_inf), mask, pto.CmpMode.EQ)
    mask_inf_src1 = pto.vcmp(src1_abs_u16, pto.vbr(f16_inf), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_inf_src0, mask_inf_src1, mask)

    # Detect Zero values
    mask_zero_src0 = pto.vcmp(src0_abs_u16, pto.vbr(pto.ui16(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src0, mask)
    mask_zero_src1 = pto.vcmp(src1_abs_u16, pto.vbr(pto.ui16(0)), mask, pto.CmpMode.EQ)
    mask_invalid = pto.por(mask_invalid, mask_zero_src1, mask)

    mask_valid = pto.pnot(mask_invalid, mask)

    # Detect subnormal numbers (denormals). Both operands use ``LT`` against the
    # smallest normal, so the whole [2^-24, 2^-14) range is re-scaled.
    mask_src0_subnormal = pto.vcmp(src0_abs_u16, pto.vbr(subnormal_threshold), mask, pto.CmpMode.LT)
    mask_src0_normal = pto.pnot(mask_src0_subnormal, mask)
    src0_subnormal = pto.vmuls(src0, normalize_scale_enlarge, mask_src0_subnormal)

    mask_src1_subnormal = pto.vcmp(src1_abs_u16, pto.vbr(subnormal_threshold), mask, pto.CmpMode.LT)
    mask_src1_normal = pto.pnot(mask_src1_subnormal, mask)
    src1_subnormal = pto.vmuls(src1, normalize_scale_enlarge, mask_src1_subnormal)

    # Merge normalized subnormals with normal values
    src0_all = pto.vsel(src0, src0_subnormal, mask_src0_normal)
    src1_all = pto.vsel(src1, src1_subnormal, mask_src1_normal)

    src0_all_u16 = pto.vbitcast(src0_all, pto.ui16)
    src1_all_u16 = pto.vbitcast(src1_all, pto.ui16)

    # Standardize exponent bits (corresponds to Div754.hpp:391-401)
    src0_norm_u16 = pto.vand(src0_all_u16, pto.vbr(exponent_extractor), mask_valid)
    src1_norm_u16 = pto.vand(src1_all_u16, pto.vbr(exponent_extractor), mask_valid)

    src0_norm_u16 = pto.vadd(src0_norm_u16, pto.vbr(exponent_normalizer), mask_valid)
    src1_norm_u16 = pto.vadd(src1_norm_u16, pto.vbr(exponent_normalizer), mask_valid)

    src0_norm_f16 = pto.vbitcast(src0_norm_u16, pto.f16)
    src1_norm_f16 = pto.vbitcast(src1_norm_u16, pto.f16)
    src0_norm = pto.vsel(src0_norm_f16, src0_all, mask_valid)
    src1_norm = pto.vsel(src1_norm_f16, src1_all, mask_valid)
    return _F16NormalizedState(
        src0_norm, src1_norm, src0_abs, src1_abs, src0_all_u16,
        src1_all_u16, mask_src0_subnormal, mask_src0_normal,
        mask_src1_subnormal, mask_src1_normal, mask_valid,
    )


def _div_f16_underflow(dst_u16, dst_sign, scale, state, mask):
    """Replace f16 denormal underflow results in the u16 domain.

    Equivalent of the f32 underflow phase with f16 bias (-9 boundary).
    Mirrors Div754.hpp:443-463.
    """
    src0_norm = state.src0_norm
    src1_norm = state.src1_norm
    mask_valid = state.mask_valid

    neg9 = pto.si16(-9)
    min_denormal = pto.ui16(0x1)
    src0_norm_abs = pto.vabs(src0_norm, mask_valid)
    src1_norm_abs = pto.vabs(src1_norm, mask_valid)
    mask_norm = pto.vcmp(src0_norm_abs, src1_norm_abs, mask_valid, pto.CmpMode.LE)
    mask_underflow1 = pto.vcmp(scale, pto.vbr(neg9), mask, pto.CmpMode.EQ)
    mask_underflow1 = pto.pand(mask_underflow1, mask_valid, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(min_denormal), mask_underflow1)
    z2_u16 = pto.vadd(dst_sign, pto.vbr(pto.ui16(0)), mask_underflow1)

    z1_sel = pto.vsel(z2_u16, z1_u16, mask_norm)
    dst_u16_temp = pto.vsel(z1_sel, dst_u16, mask_underflow1)

    mask_underflow1_not = pto.pnot(mask_underflow1, mask)
    mask_valid_temp = pto.pand(mask_underflow1_not, mask_valid, mask)

    # Underflow handling: scale < -9 (corresponds to Div754.hpp:456-463)
    mask_underflow2 = pto.vcmp(scale, pto.vbr(neg9), mask, pto.CmpMode.LT)
    mask_underflow2 = pto.pand(mask_underflow2, mask_valid_temp, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(pto.ui16(0)), mask_underflow2)
    dst_u16_temp = pto.vsel(z1_u16, dst_u16_temp, mask_underflow2)

    mask_underflow2_not = pto.pnot(mask_underflow2, mask)
    mask_valid_temp = pto.pand(mask_underflow2_not, mask_valid_temp, mask)
    return dst_u16_temp, mask_valid_temp


def _div_f16_boundary(underflow_result, dst_sign, scale, state, mask):
    """Clamp f16 overflow, rescale by the exponent and propagate NaNs.

    f16 counterpart of the f32 boundary phase with bias-15 exponent
    handling. Mirrors Div754.hpp:465-501.
    """
    dst_u16_temp, mask_valid_temp = underflow_result
    src0_abs = state.src0_abs
    src1_abs = state.src1_abs

    f16_inf = pto.ui16(0x7C00)
    nan_value = pto.ui16(0x7E00)
    max_exp = pto.si16(31)
    mask_overflow1 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.EQ)
    mask_overflow1 = pto.pand(mask_overflow1, mask_valid_temp, mask)

    scale_adj = pto.vadds(scale, pto.si16(-1), mask_overflow1)
    scale = pto.vsel(scale_adj, scale, mask_overflow1)

    dst_f16_temp = pto.vbitcast(dst_u16_temp, pto.f16)
    z1_f16 = pto.vmuls(dst_f16_temp, pto.f16(2.0), mask_overflow1)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_overflow1)

    # Overflow handling: scale > 31 (corresponds to Div754.hpp:474-480)
    mask_overflow2 = pto.vcmp(scale, pto.vbr(max_exp), mask, pto.CmpMode.GT)
    mask_overflow2 = pto.pand(mask_overflow2, mask_valid_temp, mask)

    z1_u16 = pto.vadd(dst_sign, pto.vbr(f16_inf), mask_overflow2)
    dst_u16_temp = pto.vbitcast(dst_f16_temp, pto.ui16)
    dst_u16_temp = pto.vsel(z1_u16, dst_u16_temp, mask_overflow2)

    mask_overflow2_not = pto.pnot(mask_overflow2, mask)
    mask_valid_final = pto.pand(mask_overflow2_not, mask_valid_temp, mask)

    # Positive exponent handling (corresponds to Div754.hpp:482-486)
    zero_exp = pto.si16(0)
    mask_pos_exp = pto.vcmp(scale, pto.vbr(zero_exp), mask_valid_final, pto.CmpMode.GT)

    scale_u16 = pto.vbitcast(scale, pto.ui16)
    exp_shifted = pto.vshls(scale_u16, pto.i16(10), mask_pos_exp)
    exp_factor_f16 = pto.vbitcast(exp_shifted, pto.f16)

    dst_f16_temp = pto.vbitcast(dst_u16_temp, pto.f16)
    z1_f16 = pto.vmul(dst_f16_temp, exp_factor_f16, mask_pos_exp)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_pos_exp)

    # Negative exponent handling (corresponds to Div754.hpp:488-493)
    mask_pos_exp_not = pto.pnot(mask_pos_exp, mask_valid_final)

    # Value 0x0200 = Float16 with exp=0, mantissa bit9=1 (used for shift calculation)
    shr_base = pto.i16(512)  # 0x0200
    scale_abs = pto.vabs(scale, mask_pos_exp_not)

    shr_base_vec = pto.vdup(shr_base, mask_pos_exp_not)
    shr_base_i16 = pto.vbitcast(shr_base_vec, pto.si16)
    shr_factor_i16 = pto.vshr(shr_base_i16, scale_abs, mask_pos_exp_not)
    shr_factor_f16 = pto.vbitcast(shr_factor_i16, pto.f16)

    z1_f16 = pto.vmul(dst_f16_temp, shr_factor_f16, mask_pos_exp_not)
    dst_f16_temp = pto.vsel(z1_f16, dst_f16_temp, mask_pos_exp_not)

    # NaN propagation (corresponds to Div754.hpp:495-501)
    mask_nan_src0 = pto.vcmp(src0_abs, src0_abs, mask, pto.CmpMode.NE)
    mask_nan_src1 = pto.vcmp(src1_abs, src1_abs, mask, pto.CmpMode.NE)
    mask_nan = pto.por(mask_nan_src0, mask_nan_src1, mask)

    nan_vec = pto.vbr(nan_value)
    nan_f16_vec = pto.vbitcast(nan_vec, pto.f16)
    dst_final = pto.vsel(nan_f16_vec, dst_f16_temp, mask_nan)
    return dst_final


def _div_ieee754_f16_impl(src0, src1, mask):
    """Complete IEEE 754 float16 high-precision division with subnormal handling.

    Follows pto-isa Div754.hpp:291-502 (DivIEEE754HalfImpl).

    Key differences from F32 implementation:
    - Uses LT against the smallest normal for both operands
    - Normalization factor: 2^10 (not 2^23 for F32)
    - Exponent bias: 15 (not 127 for F32)
    - Exponent shift: 10 bits (not 23 for F32)
    - Direct vdiv call (no three-candidate search)
    """
    state = _div_f16_normalize(src0, src1, mask)

    f16_inf = pto.ui16(0x7C00)
    sign_extractor = pto.ui16(0x8000)
    normalize_scale_reduce = pto.f16(0.0009765625)
    normalize_scale_enlarge = pto.f16(1024.0)
    dst = pto.vdiv(state.src0_norm, state.src1_norm, mask)

    # Subnormal dividend, normal divisor: scale down result
    # Corresponds to Div754.hpp:408-412
    mask0 = pto.pand(state.mask_src0_subnormal, state.mask_src1_normal, mask)
    z1 = pto.vmuls(dst, normalize_scale_reduce, mask0)
    dst = pto.vsel(z1, dst, mask0)

    # Normal dividend, subnormal divisor: scale up result
    # Corresponds to Div754.hpp:414-419
    mask0 = pto.pand(state.mask_src0_normal, state.mask_src1_subnormal, mask)
    z1 = pto.vmuls(dst, normalize_scale_enlarge, mask0)
    dst = pto.vsel(z1, dst, mask0)

    # Preserve sign for overflow/underflow handling
    dst_u16 = pto.vbitcast(dst, pto.ui16)
    dst_sign = pto.vand(dst_u16, pto.vbr(sign_extractor), mask)

    # Extract exponent bits (corresponds to Div754.hpp:428-439)
    src0_exponent = pto.vand(state.src0_all_u16, pto.vbr(f16_inf), mask)
    src1_exponent = pto.vand(state.src1_all_u16, pto.vbr(f16_inf), mask)

    src0_exp_shifted = pto.vshrs(src0_exponent, pto.i16(10), mask)
    src1_exp_shifted = pto.vshrs(src1_exponent, pto.i16(10), mask)

    src0_exp_i16 = pto.vbitcast(src0_exp_shifted, pto.si16)
    src1_exp_i16 = pto.vbitcast(src1_exp_shifted, pto.si16)

    scale = pto.vsub(src0_exp_i16, src1_exp_i16, mask)
    scale = pto.vadds(scale, pto.si16(15), mask)

    underflow_result = _div_f16_underflow(
        dst_u16, dst_sign, scale, state, mask,
    )
    return _div_f16_boundary(
        underflow_result, dst_sign, scale, state, mask,
    )
