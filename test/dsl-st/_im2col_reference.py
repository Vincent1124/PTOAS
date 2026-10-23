# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared im2col (A5 LOAD L1 TO L0A 3Dv2) reference helpers for the DSL ST cases.

The hardware loads an NC1HWC0 image and flattens each sliding-window position
into one im2col row; each output row is one window position in (ho, wo) order,
each column is the window flattened in (kh, kw, c) order.  The B operand is
loaded into L0B transposed, so the reference matmul is
``col(M x K) @ B(K x N)`` with ``B = w.transpose(1, 2, 3, 0).reshape(K, N)``.
"""

import numpy as np


def make_fmap(fmap_h, fmap_w, channel, dtype=np.float16) -> np.ndarray:
    """Deterministic fmap: (fmap_h, fmap_w, channel), values in a small range."""
    h = np.arange(fmap_h, dtype=np.float32).reshape(fmap_h, 1, 1)
    w = np.arange(fmap_w, dtype=np.float32).reshape(1, fmap_w, 1)
    c = np.arange(channel, dtype=np.float32).reshape(1, 1, channel)
    raw = np.mod(h * 7.0 + w * 3.0 + c, 13.0) - 6.0
    return (raw / np.float32(4.0)).astype(dtype)


def make_weight(out_ch, channel, filter_h, filter_w, dtype=np.float16) -> np.ndarray:
    """Deterministic weights: (out_ch, channel, filter_h, filter_w)."""
    n = np.arange(out_ch, dtype=np.float32).reshape(out_ch, 1, 1, 1)
    c = np.arange(channel, dtype=np.float32).reshape(1, channel, 1, 1)
    kh = np.arange(filter_h, dtype=np.float32).reshape(1, 1, filter_h, 1)
    kw = np.arange(filter_w, dtype=np.float32).reshape(1, 1, 1, filter_w)
    raw = np.mod(n * 5.0 + c * 2.0 + kh * 3.0 + kw, 11.0) - 5.0
    return (raw / np.float32(4.0)).astype(dtype)


def make_b(weight, channel, filter_h, filter_w, out_ch) -> np.ndarray:
    """B operand layout as loaded into L0B with transpose: (K, N).

    The reference im2col columns are flattened in (kh, kw, c) order, so the
    matching B rows must be in the same order: row k = (kh, kw, c) pairs with
    ``weight[n, c, kh, kw]``.  A channel-major layout (c, kh, kw) would pair
    mismatched elements and produce a non-convolution product.
    """
    return weight.transpose(2, 3, 1, 0).reshape(channel * filter_h * filter_w, out_ch)


def im2col_conv_reference(
    a,
    b,
    *,
    fmap_h,
    fmap_w,
    filter_h,
    filter_w,
    channel,
    out_ch,
    stride_h=1,
    stride_w=1,
    dilation_h=1,
    dilation_w=1,
    pad_top=0,
    pad_bottom=0,
    pad_left=0,
    pad_right=0,
) -> np.ndarray:
    """Reference im2col convolution; returns the (M, N) f32 golden.

    ``b`` is the (K, N) B operand in the L0B-transposed layout (see
    :func:`make_b`); the reference matmul is ``col(M x K) @ b(K x N)``.
    """
    img = np.pad(
        a.astype(np.float32),
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
    )
    eff_h = (filter_h - 1) * dilation_h + 1
    eff_w = (filter_w - 1) * dilation_w + 1
    out_h = (img.shape[0] - eff_h) // stride_h + 1
    out_w = (img.shape[1] - eff_w) // stride_w + 1
    windows = []
    for ho in range(out_h):
        for wo in range(out_w):
            patch = img[
                ho * stride_h : ho * stride_h + eff_h : dilation_h,
                wo * stride_w : wo * stride_w + eff_w : dilation_w,
                :,
            ]
            windows.append(patch.reshape(-1))
    col = np.stack(windows)  # (M, K), column order (kh, kw, c)
    return (col @ b.astype(np.float32)).astype(np.float32)
