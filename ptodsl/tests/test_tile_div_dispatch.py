#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Tile division dispatch must share the SoftOps implementations.

The tile path historically carried its own f32 IEEE-754 implementation, which
classified subnormal dividends with an equality test against the largest
subnormal. Routing through SoftOps keeps one implementation per algorithm, so the
tile path and the VMI path cannot drift apart.
"""

import pytest

from SoftOps.div_vector import (
    div_vf32_soft,
    div_vf32_ulp0_ftz_false,
    div_vf32_ulp1_ftz_false,
)
from TileOps.a5.elementwise import div_dispatch

F32_SOFTWARE_CASES = {
    "diff_compensation": div_vf32_soft,
    "precision_0ulp_ftz_true": div_vf32_soft,
    "precision_0ulp_ftz_false": div_vf32_ulp0_ftz_false,
    "precision_1ulp_ftz_false": div_vf32_ulp1_ftz_false,
}

ZERO_ULP_ALGOS = ("precision_0ulp_ftz_true", "precision_0ulp_ftz_false")


def test_f32_table_matches_the_softops_implementations():
    """No tile-local f32 implementation: the table holds the SoftOps callables."""
    assert div_dispatch.F32_SOFTWARE == F32_SOFTWARE_CASES


@pytest.mark.parametrize("algo", sorted(F32_SOFTWARE_CASES))
def test_f32_dispatch_routes_to_the_shared_helper(algo, monkeypatch):
    """``divide`` must call the SoftOps helper selected by the algorithm."""
    seen = {}

    def fake_divide(lhs, rhs, mask):
        seen.update(lhs=lhs, rhs=rhs, mask=mask, algo=algo)
        return "soft-result"

    monkeypatch.setitem(div_dispatch.F32_SOFTWARE, algo, fake_divide)
    assert div_dispatch.divide("lhs", "rhs", "f32", "mask", algo) == "soft-result"
    assert seen["algo"] == algo


@pytest.mark.parametrize("algo", ZERO_ULP_ALGOS)
def test_f16_rejects_the_zero_ulp_algorithms(algo):
    """f16 has no 0-ULP implementation, matching the AscendC static_assert."""
    with pytest.raises(ValueError, match="f32-only"):
        div_dispatch.divide(None, None, "f16", None, algo)


def test_f16_one_ulp_ftz_false_uses_the_half_implementation(monkeypatch):
    """The only f16 software algorithm is DivIEEE754HalfImpl."""
    implementation = div_dispatch.F16_SOFTWARE["precision_1ulp_ftz_false"]
    assert implementation.__name__ == "_div_ieee754_f16_impl"
    monkeypatch.setitem(div_dispatch.F16_SOFTWARE,
                        "precision_1ulp_ftz_false", lambda l, r, m: "half")
    assert div_dispatch.divide("lhs", "rhs", "f16", "mask",
                               "precision_1ulp_ftz_false") == "half"
