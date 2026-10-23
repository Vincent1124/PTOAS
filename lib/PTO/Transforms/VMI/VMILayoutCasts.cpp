// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Layout relation enumeration: casts.
#include "VMILayoutRelations.h"
#include "PTO/Transforms/VMILayoutPropagation.h"
#include "llvm/ADT/TypeSwitch.h"
namespace mlir::pto::layout_relations {

void VMILayoutRelations::appendCastFact(const VMICastLayoutFact& fact, const FailureOr<VMICastLayoutFact>& preferred)
{
    appendReachableUniqueRelation(
        relations,
        VMILayoutOpRelation{
            op,
            {operandPort(0, fact.sourceLayout), resultPort(0, fact.resultLayout)},
            false,
            fact.intrinsicRearrangementCost,
            getPairPreferencePenalty(fact, preferred)},
        supports);
}

void VMILayoutRelations::appendSourceCastFacts(const CastRelationState& state, VMIVRegType concreteSourceType)
{
    auto [sourceType, resultType, sameWidthNumericCast, preferred] = state;
    (void)sourceType;

    if (sameWidthNumericCast) {
        VMILayoutAttr layout = concreteSourceType.getLayoutAttr();
        if (!layout || !isLegalLayoutForType(resultType, layout)) {
            return;
        }
        VMIVRegType concreteResultType = VMIVRegType::get(
            resultType.getContext(), resultType.getElementCount(), resultType.getElementType(), layout);
        auto fact = supports.getSameWidthCastLayoutFact(concreteSourceType, concreteResultType);
        if (succeeded(fact)) {
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{op, {operandPort(0, fact->sourceLayout), resultPort(0, fact->resultLayout)}, false},
                supports);
        }
        return;
    }
    auto facts = supports.getCastLayoutFacts(concreteSourceType, resultType);
    if (failed(facts)) {
        return;
    }
    for (const VMICastLayoutFact& fact : *facts) {
        if (failed(supports.validateCastOperationRelation(op, fact.sourceLayout, fact.resultLayout))) {
            continue;
        }
        appendCastFact(fact, preferred);
    }
}
void VMILayoutRelations::appendAnnotatedCastFacts(const CastRelationState& state)
{
    auto [sourceType, resultType, sameWidthNumericCast, preferred] = state;
    (void)preferred;
    VMILayoutAttr annotatedSource = sourceType.getLayoutAttr();
    VMILayoutAttr annotatedResult = resultType.getLayoutAttr();
    if (!annotatedSource || !annotatedResult) {
        return;
    }
    bool annotatedPairIsLive = sameWidthNumericCast;
    if (!annotatedPairIsLive) {
        std::string annotatedReason;
        annotatedPairIsLive = succeeded(supports.getCastLayoutFactForLayouts(
            sourceType, resultType, annotatedSource, annotatedResult, &annotatedReason));
    }
    if (!annotatedPairIsLive) {
        return;
    }
    // The pair the IR states is a relation like any other, so it carries the
    // same facts - in particular the cast's intrinsic rearrangement cost,
    // which the physical graph can never see because a cast lowers as a
    // single operation.  Pricing it as free would make a group-slot
    // extension that unpacks every physical part look cheaper than any
    // alternative, so the pair is priced through the same enumerating query
    // the rows below use, and a pair that cannot be priced is not offered.
    FailureOr<SmallVector<VMICastLayoutFact, mlir::pto::kValue4>> priced = supports.getCastLayoutFactsForLayout(
        sourceType, resultType, VMICastLayoutPort::Source, annotatedSource);
    if (failed(priced)) {
        return;
    }
    for (const VMICastLayoutFact& fact : *priced) {
        if (fact.resultLayout != annotatedResult) {
            continue;
        }
        // The pair goes in first - it is the one the IR states - but the
        // other rows stay candidates: they are what the tables enumerate,
        // and a relation set that hid them would make the solver's own view
        // and the conformance view disagree.  A pair that is *live* is not
        // automatically the cheapest any more either, now that it carries
        // its intrinsic cost, which is the point of pricing it.
        // The pair keeps preference penalty 0 even though it now carries its
        // intrinsic cost: it is the pair the IR states, so it must not be
        // out-competed on a preference key, and the enumerated rows below
        // may reach the same pair through the table with penalty 1.  The two
        // differ only in that key and select the same layouts, so the extra
        // entry costs nothing but is visible in the relation listing; making
        // the penalties agree instead (so they collapse) regressed
        // vmi_layout_assignment_group_slot_load, which needs the stated pair
        // to keep priority.
        appendReachableUniqueRelation(
            relations,
            VMILayoutOpRelation{
                op,
                {operandPort(0, annotatedSource), resultPort(0, annotatedResult)},
                false,
                fact.intrinsicRearrangementCost,
                0},
            supports);
    }
}
void VMILayoutRelations::appendResultCastFacts(const CastRelationState& state)
{
    auto [sourceType, resultType, sameWidthNumericCast, preferred] = state;
    (void)sameWidthNumericCast;
    VMILayoutAttr annotatedResult = resultType.getLayoutAttr();

    SmallVector<VMILayoutAttr, mlir::pto::kValue4> resultCandidates;
    rememberLayout(annotatedResult, resultCandidates);
    for (VMILayoutAttr layout : polymorphicLayouts) {
        rememberLayout(layout, resultCandidates);
    }
    for (VMILayoutAttr resultLayout : resultCandidates) {
        if (!isLegalLayoutForType(resultType, resultLayout)) {
            continue;
        }
        auto facts =
            supports.getCastLayoutFactsForLayout(sourceType, resultType, VMICastLayoutPort::Result, resultLayout);
        if (failed(facts)) {
            continue;
        }
        for (const VMICastLayoutFact& fact : *facts) {
            if (!annotatedResult &&
                failed(supports.validateCastOperationRelation(op, fact.sourceLayout, fact.resultLayout))) {
                continue;
            }
            appendCastFact(fact, preferred);
        }
    }
}
void VMILayoutRelations::appendSameWidthCastFacts(const CastRelationState& state)
{
    auto candidates = getDenseCandidateLayouts(state.resultType.getLayoutAttr(), polymorphicLayouts, op->getContext());
    for (VMILayoutAttr layout : candidates) {
        if (!isLegalLayoutForType(state.sourceType, layout)) {
            continue;
        }
        auto concreteSourceType = VMIVRegType::get(
            state.sourceType.getContext(), state.sourceType.getElementCount(), state.sourceType.getElementType(),
            layout);
        appendSourceCastFacts(state, concreteSourceType);
    }
}
RelationResult VMILayoutRelations::buildCast()
{
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return failure();
    }
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !resultType) {
        return failure();
    }
    bool sameWidthNumericCast = isa<VMIFPToSIOp, VMIFPToUIOp, VMISIToFPOp>(op) &&
                                pto::getPTOStorageElemBitWidth(sourceType.getElementType()) != 0 &&
                                pto::getPTOStorageElemBitWidth(sourceType.getElementType()) ==
                                    pto::getPTOStorageElemBitWidth(resultType.getElementType());
    CastRelationState state{
        sourceType, resultType, sameWidthNumericCast, supports.getPreferredCastLayoutFact(sourceType, resultType)};

    appendSourceCastFacts(state, sourceType);
    // Casts whose producer has no explicit layout still need a contiguous
    // seed so the cost solver can materialize their result into a downstream
    // relation when no table row uses the component's first candidate.
    if (relations.empty() && !sourceType.getLayoutAttr()) {
        auto contiguousSourceType = VMIVRegType::get(
            sourceType.getContext(), sourceType.getElementCount(), sourceType.getElementType(),
            VMILayoutAttr::getContiguous(op->getContext()));
        appendSourceCastFacts(state, contiguousSourceType);
    }
    SmallVector<int64_t, mlir::pto::kValue2> instantiatedGroups;
    for (VMILayoutAttr layout : polymorphicLayouts) {
        if (!layout || !layout.isGroupSlots() || !isLegalLayoutForType(sourceType, layout) ||
            llvm::is_contained(instantiatedGroups, layout.getNumGroups())) {
            continue;
        }
        instantiatedGroups.push_back(layout.getNumGroups());
        auto concreteSourceType = VMIVRegType::get(
            sourceType.getContext(), sourceType.getElementCount(), sourceType.getElementType(), layout);
        appendSourceCastFacts(state, concreteSourceType);
    }
    appendAnnotatedCastFacts(state);

    if (sameWidthNumericCast) {
        // Same-width numeric casts preserve layout. Query the same-width
        // support facts for dense candidates from either side, including an
        // explicit result whose source is still unassigned.
        appendSameWidthCastFacts(state);
    }

    // A cast is also reachable from its result: the pass' own cast transfer
    // queries the table for whichever side changed, so an annotated result
    // states which rows apply.  Those rows are the IR's own consequence rather
    // than a solver choice, so the solver's gate is not applied to them.
    if (!sameWidthNumericCast) {
        appendResultCastFacts(state);
    }
    if (relations.empty()) {
        return failure();
    }
    return relations;
}

RelationResult VMILayoutRelations::buildEnsure()
{
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return failure();
    }
    VMILayoutAttr sourceLayout = getVMIExplicitLayout(op->getOperand(0).getType());
    VMILayoutAttr resultLayout = getVMIExplicitLayout(op->getResult(0).getType());
    if (!sourceLayout || !resultLayout) {
        return failure();
    }
    LogicalResult supported = failure();
    if (auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType())) {
        auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
        if (resultType) {
            supported = success(succeeded(supports.getEnsureLayoutFact(sourceType, resultType)));
        }
    } else if (auto sourceType = dyn_cast<VMIMaskType>(op->getOperand(0).getType())) {
        auto resultType = dyn_cast<VMIMaskType>(op->getResult(0).getType());
        if (resultType) {
            supported = success(succeeded(supports.getEnsureMaskLayoutFact(sourceType, resultType)));
        }
    }
    if (failed(supported)) {
        return failure();
    }
    return makeReachableRelation({op, {operandPort(0, sourceLayout), resultPort(0, resultLayout)}, false}, supports);
}

static SmallVector<VMILayoutAttr, kValue4> getMaskGranularityCandidates(
    VMIMaskType type, ArrayRef<VMILayoutAttr> seeds, bool seedContiguous)
{
    if (auto layout = type.getLayoutAttr()) {
        return {layout};
    }
    SmallVector<VMILayoutAttr, kValue4> candidates;
    if (seedContiguous) {
        candidates.push_back(VMILayoutAttr::getContiguous(type.getContext()));
    }
    for (VMILayoutAttr layout : seeds) {
        rememberLayout(layout, candidates);
    }
    return candidates;
}
RelationResult VMILayoutRelations::buildMaskGranularity(VMIEnsureMaskGranularityOp ensure)
{
    auto source = dyn_cast<VMIMaskType>(ensure.getSource().getType());
    auto result = dyn_cast<VMIMaskType>(ensure.getResult().getType());
    if (!source || !result) {
        return failure();
    }
    VMILayoutAttr explicitSource = source.getLayoutAttr();
    VMILayoutAttr explicitResult = result.getLayoutAttr();
    auto sourceCandidates = getMaskGranularityCandidates(source, polymorphicLayouts, true);
    // Asymmetric rows can be reachable only from the consumer's result layout.
    auto resultCandidates = getMaskGranularityCandidates(result, polymorphicLayouts, false);
    SmallVector<VMILayoutOpRelation, mlir::pto::kValue4> relations;
    for (VMICastLayoutPort port : {VMICastLayoutPort::Source, VMICastLayoutPort::Result}) {
        ArrayRef<VMILayoutAttr> candidates = port == VMICastLayoutPort::Source ? sourceCandidates : resultCandidates;
        for (VMILayoutAttr layout : candidates) {
            auto facts = supports.getMaskGranularityCastLayoutFactsForLayout(source, result, port, layout);
            if (failed(facts)) {
                continue;
            }
            for (const auto& fact : *facts) {
                bool matchesOtherSide = port == VMICastLayoutPort::Source ?
                                            (!explicitResult || fact.resultLayout == explicitResult) :
                                            (!explicitSource || fact.sourceLayout == explicitSource);
                if (matchesOtherSide) {
                    appendReachableUniqueRelation(
                        relations,
                        VMILayoutOpRelation{
                            op,
                            {operandPort(0, fact.sourceLayout), resultPort(0, fact.resultLayout)},
                            false,
                            fact.intrinsicRearrangementCost},
                        supports);
                }
            }
        }
    }
    return relations.empty() ? FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(failure()) :
                               FailureOr<SmallVector<VMILayoutOpRelation, mlir::pto::kValue4>>(std::move(relations));
}

RelationResult VMILayoutRelations::buildBitcast()
{
    auto sourceType = dyn_cast<VMIVRegType>(op->getOperand(0).getType());
    auto resultType = dyn_cast<VMIVRegType>(op->getResult(0).getType());
    if (!sourceType || !resultType) {
        return failure();
    }
    auto candidates = getDenseCandidateLayouts(sourceType.getLayoutAttr(), polymorphicLayouts, op->getContext());
    for (VMILayoutAttr layout : candidates) {
        auto facts = supports.getBitcastLayoutFactsForLayout(sourceType, resultType, VMICastLayoutPort::Source, layout);
        if (failed(facts)) {
            continue;
        }
        for (const VMIBitcastLayoutFact& fact : *facts) {
            appendReachableUniqueRelation(
                relations,
                VMILayoutOpRelation{op, {operandPort(0, fact.sourceLayout), resultPort(0, fact.resultLayout)}, false},
                supports);
        }
    }
    return relations;
}

} // namespace mlir::pto::layout_relations
