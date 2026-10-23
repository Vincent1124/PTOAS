// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVdiv.cpp - pto.Vdiv methods ------------------------------------===//
//===----------------------------------------------------------------------===//

#include "../VPTOVecOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::vecop_detail;

LogicalResult VdivOp::verify() {
  if (failed(verifyBinaryVecOp(*this))) {
    return failure();
  }
  const DivPrecision algo = getPrecisionType();
  const bool pureSoftware = isSoftwareDivAlgo(algo);
  if (!pureSoftware && algo != DivPrecision::DiffCompensation) {
    // The remaining selectors lower to a single native vector divide, so they
    // accept every element type the op itself supports.
    return success();
  }
  const std::string name = getDivAlgoToken(algo).str();
  auto type = cast<VRegType>(getResult().getType());
  if (pureSoftware && !type.getElementType().isF32()) {
    return emitOpError(name + " requires f32 vector operands");
  }
  if (!type.getElementType().isF32()) {
    // diff_compensation is compensated only for f32; other types stay native.
    return success();
  }
  const bool validMask =
      cast<MaskType>(getMask().getType()).getGranularity() == "b32";
  if (!validMask) {
    return emitOpError(name + " f32 requires a b32 mask");
  }
  return success();
}
