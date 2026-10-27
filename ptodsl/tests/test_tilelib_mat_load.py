# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression coverage for the logical DN shape used by GM-to-MAT loads."""

import unittest

from ptodsl.tilelib import NoMatchingTemplate, ScalarType, TileSpec, ViewSpec, select


def dn_specs(*, shape=(1, 1, 1, 64, 128), valid_shape=None, dtype="f16", column_stride=64):
    return {
        "src": ViewSpec(
            shape=shape,
            dtype=ScalarType(dtype),
            strides=(128 * column_stride,) * 3 + (1, column_stride),
        ),
        "dst": TileSpec(
            shape=(64, 128),
            valid_shape=valid_shape,
            dtype=ScalarType(dtype),
            memory_space="mat",
            b_layout="col_major",
            s_layout="row_major",
        ),
    }


class MatLoadTest(unittest.TestCase):
    def test_rectangular_dn_uses_logical_rows_and_columns(self):
        for dtype in ("f16", "bf16", "f32", "i8", "f8e4m3", "f8e5m2", "hif8"):
            for column_stride in (64, 80):
                with self.subTest(dtype=dtype, column_stride=column_stride):
                    template = select(
                        "pto.tload", "a5", dn_specs(dtype=dtype, column_stride=column_stride)
                    )
                    self.assertEqual(template.name, "template_tload_gm_to_mat_dn2nz")

    def test_padded_tile_and_valid_window(self):
        for shape, valid in (
            ((1, 1, 1, 63, 127), (63, 127)),
            ((1, 1, 1, 64, 128), (51, 123)),
        ):
            with self.subTest(shape=shape, valid=valid):
                template = select(
                    "pto.tload", "a5", dn_specs(shape=shape, valid_shape=valid)
                )
                self.assertEqual(template.name, "template_tload_gm_to_mat_dn2nz")

    def test_invalid_matrix_extents_and_batches_are_rejected(self):
        for shape in (
            (1, 1, 1, 128, 64),  # old, transposed convention
            (1, 1, 1, 32, 128),  # source does not cover valid rows
            (1, 1, 1, 64, 64),   # source does not cover valid columns
            (1, 1, 1, 128, 128),  # source rows exceed physical tile
            (1, 1, 1, 64, 256),  # source columns exceed physical tile
            (2, 1, 1, 64, 128),  # template only emits a single matrix
            (1, 2, 1, 64, 128),
            (1, 1, 2, 64, 128),
        ):
            with self.subTest(shape=shape):
                with self.assertRaises(NoMatchingTemplate):
                    select("pto.tload", "a5", dn_specs(shape=shape))

    def test_square_matrix_selection_follows_contiguous_axis(self):
        for strides, expected in (
            ((4096, 4096, 4096, 64, 1), "nd2nz"),
            ((4096, 4096, 4096, 1, 64), "dn2nz"),
        ):
            specs = dn_specs(shape=(1, 1, 1, 64, 64))
            specs["src"] = ViewSpec(
                shape=(1, 1, 1, 64, 64), dtype=ScalarType("f16"), strides=strides
            )
            specs["dst"] = TileSpec(
                shape=(64, 64), dtype=ScalarType("f16"), memory_space="mat",
                b_layout="col_major", s_layout="row_major",
            )
            with self.subTest(expected=expected):
                self.assertEqual(
                    select("pto.tload", "a5", specs).name,
                    "template_tload_gm_to_mat_" + expected,
                )


if __name__ == "__main__":
    unittest.main()
