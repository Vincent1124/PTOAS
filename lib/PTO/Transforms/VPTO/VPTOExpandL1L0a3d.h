// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOExpandL1L0a3d.h - A5 3D im2col load expansion helpers ----------===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VPTO_VPTOEXPANDL1L0A3D_H
#define PTO_TRANSFORMS_VPTO_VPTOEXPANDL1L0A3D_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace pto {

class MteL1L0a3dOp;

// Derived per-load controls produced by the 3D expansion (m_step, k_step,
// dst_stride, repeat_source_stride), either folded statically from constant
// geometry or built as an arith chain for dynamic (SSA) inputs.
struct Derived3dLoadControls {
  Value mStep;
  Value kStep;
  Value dstStride;
  Value repeatSourceStride;
};

// Folds every geometry/control operand; returns true when all are constants.
bool fold3dGeometryControls(MteL1L0a3dOp op, int64_t &fmapH, int64_t &fmapW,
                            int64_t &filterH, int64_t &filterW,
                            int64_t &channelSize, int64_t &strideH,
                            int64_t &strideW, int64_t &dilationH,
                            int64_t &dilationW, int64_t &padTop,
                            int64_t &padBottom, int64_t &padLeft,
                            int64_t &padRight, int64_t &repeatTime,
                            int64_t &repeatMode);

// Static geometry path: folds the derived controls to constants.
Derived3dLoadControls deriveStatic3dControls(
    PatternRewriter &rewriter, Location loc, MteL1L0a3dOp op, uint64_t c0Size,
    int64_t fmapH, int64_t fmapW, int64_t filterH, int64_t filterW,
    int64_t channelSize, int64_t strideH, int64_t strideW, int64_t dilationH,
    int64_t dilationW, int64_t padTop, int64_t padBottom, int64_t padLeft,
    int64_t padRight, int64_t repeatTime, int64_t repeatMode);

// Dynamic geometry path: builds the arith chain for the derived controls.
Derived3dLoadControls deriveDynamic3dControls(PatternRewriter &rewriter,
                                              Location loc, MteL1L0a3dOp op,
                                              uint64_t c0Size);

// Packs the set_fmatrix register control word from the geometry values.
Value packL3dFmatrix(PatternRewriter &rewriter, Location loc, Value fmapW,
                     Value fmapH, Value padLeft, Value padRight, Value padTop,
                     Value padBottom);

// Packs the set_l3d_rpt register control word from the repeat controls.
Value packL3dRepeat(PatternRewriter &rewriter, Location loc,
                    Value repeatSourceStride, Value repeatTime,
                    Value repeatMode, Value dstStride, Value dstMPosition);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_VPTO_VPTOEXPANDL1L0A3D_H
