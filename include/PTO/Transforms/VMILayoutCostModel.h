// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under
// the terms and conditions of CANN Open Software License Agreement Version 2.0
// (the "License"). Please refer to the License for details. You may not use
// this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
// AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
// FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
// for the full text of the License.

//===- VMILayoutCostModel.h - VMI physical layout cost model -*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef PTO_TRANSFORMS_VMILAYOUTCOSTMODEL_H
#define PTO_TRANSFORMS_VMILAYOUTCOSTMODEL_H

#include "PTO/Transforms/VMILayoutPhysicalState.h"

namespace mlir::pto {

struct VMILayoutScopeCost {
    int64_t total = 0;
    // Loop-weighted physical memory actions. Compared after total: a unit
    // memory action is not a calibrated substitute for arithmetic or shuffles.
    int64_t memoryTotal = 0;
    // Intrinsic rearrangements performed inside a compound lowering, such as
    // the zero/source interleave used by same-layout BF16 widening.
    int64_t intrinsicRearrangements = 0;
    // Number of Merge actions in the plan.  These are the "vor"/"por" merges
    // that join two rearranged halves.
    int64_t merges = 0;
    // Per-kind action counts, kept for diagnostics and for the tie-break layers.
    int64_t memoryReads = 0;
    int64_t memoryWrites = 0;
    int64_t semantics = 0;
    int64_t deinterleaves = 0;
    int64_t unpacks = 0;
    int64_t packs = 0;
    // Number of Interleave actions in the plan.  An interleave ("vintlv") also
    // rejoins data that a layout spread over separate registers, so the solver
    // counts it together with a merge.
    int64_t interleaves = 0;
};

FailureOr<VMILayoutPhysicalState> appendVMILayoutPhysicalRelation(
    const VMILayoutPhysicalState& state, const VMILayoutOpRelation& relation, const VMILayoutPlan& partialPlan);

FailureOr<std::string> getVMILayoutContinuationKey(
    const VMILayoutPhysicalState& state, ArrayRef<Operation*> remainingOps);

VMILayoutScopeCost getVMILayoutPhysicalCost(const VMILayoutPhysicalState& state);

FailureOr<VMILayoutScopeCost> evaluateVMILayoutPlanCost(
    ArrayRef<VMILayoutOpRelation> selectedRelations, const VMILayoutPlan& plan);

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_VMILAYOUTCOSTMODEL_H
