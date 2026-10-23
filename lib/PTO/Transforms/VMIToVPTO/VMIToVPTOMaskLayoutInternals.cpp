// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Predicate layout/granularity materializers. Textual unit of VMIToVPTO.cpp;
// included after data layout helpers and before operation patterns.
#pragma once
enum class PredicateInterleaveKind { Interleave, Deinterleave };

static FailureOr<std::pair<Value, Value>> createPredicateInterleave(
    Location loc, Type lowType, Type highType, Value lhs, Value rhs, PredicateInterleaveKind kind,
    PatternRewriter& rewriter)
{
    auto maskType = dyn_cast<MaskType>(lowType);
    if (!maskType || highType != lowType) {
        return failure();
    }
    bool deinterleave = kind == PredicateInterleaveKind::Deinterleave;
    if (maskType.isB8()) {
        if (deinterleave) {
            auto op = rewriter.create<PdintlvB8Op>(loc, lowType, highType, lhs, rhs);
            return std::make_pair(op.getLow(), op.getHigh());
        }
        auto op = rewriter.create<PintlvB8Op>(loc, lowType, highType, lhs, rhs);
        return std::make_pair(op.getLow(), op.getHigh());
    }
    if (maskType.isB16()) {
        if (deinterleave) {
            auto op = rewriter.create<PdintlvB16Op>(loc, lowType, highType, lhs, rhs);
            return std::make_pair(op.getLow(), op.getHigh());
        }
        auto op = rewriter.create<PintlvB16Op>(loc, lowType, highType, lhs, rhs);
        return std::make_pair(op.getLow(), op.getHigh());
    }
    if (maskType.isB32()) {
        if (deinterleave) {
            auto op = rewriter.create<PdintlvB32Op>(loc, lowType, highType, lhs, rhs);
            return std::make_pair(op.getLow(), op.getHigh());
        }
        auto op = rewriter.create<PintlvB32Op>(loc, lowType, highType, lhs, rhs);
        return std::make_pair(op.getLow(), op.getHigh());
    }
    return failure();
}

FailureOr<std::pair<Value, Value>> createPredicateDintlv(
    Location loc, Type lowType, Type highType, Value lhs, Value rhs, PatternRewriter& rewriter)
{
    return createPredicateInterleave(loc, lowType, highType, lhs, rhs, PredicateInterleaveKind::Deinterleave, rewriter);
}

FailureOr<std::pair<Value, Value>> createPredicateIntlv(
    Location loc, Type lowType, Type highType, Value lhs, Value rhs, PatternRewriter& rewriter)
{
    return createPredicateInterleave(loc, lowType, highType, lhs, rhs, PredicateInterleaveKind::Interleave, rewriter);
}

static FailureOr<SmallVector<Value>> materializeDeinterleaved2MaskToContiguous(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, PatternRewriter& rewriter)
{
    int64_t groups = sourceParts.size() / kVMIDataLayoutFactor2;
    SmallVector<Value> results;
    results.reserve(sourceParts.size());
    for (int64_t i = 0; i < groups; ++i) {
        FailureOr<std::pair<Value, Value>> materialize = createPredicateIntlv(
            op->getLoc(), resultTypes[2 * i], resultTypes[2 * i + 1], sourceParts[i], sourceParts[groups + i],
            rewriter);
        if (failed(materialize)) {
            return rewriter.notifyMatchFailure(op, "unsupported predicate intlv mask type");
        }
        results.append({materialize->first, materialize->second});
    }
    return results;
}

static FailureOr<SmallVector<Value>> materializeContiguousToDeinterleaved2Mask(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, PatternRewriter& rewriter)
{
    // One deinterleaved part per pair of result types.  A sub-chunk contiguous
    // source (a single part the 2-way split expands into two) has no second
    // source part, so its second predicate operand is the all-false mask: the
    // deinterleave then keeps only the even/odd image of the sole part and zeroes
    // the duplicated half instead of marking that redundant half valid.
    int64_t groups = resultTypes.size() / kVMIDataLayoutFactor2;
    SmallVector<Value> part0;
    SmallVector<Value> part1;
    part0.reserve(groups);
    part1.reserve(groups);
    for (int64_t i = 0; i < groups; ++i) {
        size_t lhsIndex = 2 * i;
        if (lhsIndex >= sourceParts.size()) {
            return rewriter.notifyMatchFailure(
                op, "contiguous to deinterleaved=2 mask materialization missing "
                    "source part");
        }
        Value rhs = sourceParts[lhsIndex];
        if (lhsIndex + 1 < sourceParts.size()) {
            rhs = sourceParts[lhsIndex + 1];
        } else {
            auto maskType = dyn_cast<MaskType>(resultTypes[i]);
            if (!maskType) {
                return rewriter.notifyMatchFailure(
                    op, "contiguous to deinterleaved=2 mask materialization requires a "
                        "predicate result");
            }
            FailureOr<Value> allFalse = createPrefixMask(op->getLoc(), maskType, "PAT_ALLF", rewriter);
            if (failed(allFalse)) {
                return rewriter.notifyMatchFailure(op, "failed to create all-false deinterleave mask");
            }
            rhs = *allFalse;
        }
        FailureOr<std::pair<Value, Value>> materialize = createPredicateDintlv(
            op->getLoc(), resultTypes[i], resultTypes[groups + i], sourceParts[lhsIndex], rhs, rewriter);
        if (failed(materialize)) {
            return rewriter.notifyMatchFailure(op, "unsupported predicate dintlv mask type");
        }
        part0.push_back(materialize->first);
        part1.push_back(materialize->second);
    }
    SmallVector<Value> results;
    results.append(part0);
    results.append(part1);
    return results;
}

enum class Deinterleaved2MaskLayoutDirection { Unsupported, ToContiguous, FromContiguous };

static Deinterleaved2MaskLayoutDirection getDeinterleaved2MaskDirection(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout)
{
    bool sourceIsDeinterleaved2 = sourceLayout && sourceLayout.isDeinterleaved() && sourceLayout.getFactor() == 2 &&
                                  sourceLayout.getLaneStride() == 1;
    bool resultIsDeinterleaved2 = resultLayout && resultLayout.isDeinterleaved() && resultLayout.getFactor() == 2 &&
                                  resultLayout.getLaneStride() == 1;
    bool toContiguous =
        sourceIsDeinterleaved2 && resultLayout && resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    bool fromContiguous =
        sourceLayout && sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 && resultIsDeinterleaved2;
    if (toContiguous) {
        return Deinterleaved2MaskLayoutDirection::ToContiguous;
    }
    if (fromContiguous) {
        return Deinterleaved2MaskLayoutDirection::FromContiguous;
    }
    return Deinterleaved2MaskLayoutDirection::Unsupported;
}

static FailureOr<std::optional<SmallVector<Value>>> materializeDeinterleaved2MaskLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    Deinterleaved2MaskLayoutDirection direction = getDeinterleaved2MaskDirection(sourceLayout, resultLayout);
    if (direction == Deinterleaved2MaskLayoutDirection::Unsupported) {
        return std::optional<SmallVector<Value>>{};
    }
    // The reverse (deinterleaved -> contiguous) move is the full 2*N -> 2*N
    // relation, but the forward move also covers a sub-chunk contiguous source
    // (one part expanding into two) whose halves live in the two halves of the
    // same physical carrier.  That case is not an identity forward, so it skips
    // the per-part type check and lets pdintlv reorder the predicate lanes.
    bool equalArity = sourceParts.size() == resultTypes.size();
    bool subChunkForward = direction == Deinterleaved2MaskLayoutDirection::FromContiguous &&
                           sourceParts.size() * kVMIDataLayoutFactor2 == resultTypes.size();
    // An unequal part count that the sub-chunk forward relation does not
    // describe is not a per-part rearrangement: the grouped staging
    // materializers consume a grouped source and produce the grouped result
    // arity, so route there instead of refusing.
    if (!equalArity && !subChunkForward) {
        return direction == Deinterleaved2MaskLayoutDirection::ToContiguous ?
                   materializeStagingDeintToContiguousMaskLayout(
                       op, sourceParts, resultTypes, kVMIDataLayoutFactor2, rewriter) :
                   materializeStagingContiguousToDeintMaskLayout(
                       op, sourceParts, resultTypes, kVMIDataLayoutFactor2, rewriter);
    }
    bool invalidArity = sourceParts.empty() || resultTypes.empty() || (!equalArity && !subChunkForward) ||
                        resultTypes.size() % kVMIDataLayoutFactor2 != 0;
    if (invalidArity) {
        (void)rewriter.notifyMatchFailure(
            op, "deinterleaved=2 mask layout materialization requires 2*N parts "
                "or a single contiguous source part");
        return failure();
    }
    if (equalArity && failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes, rewriter))) {
        return failure();
    }

    if (direction == Deinterleaved2MaskLayoutDirection::ToContiguous) {
        FailureOr<SmallVector<Value>> results =
            materializeDeinterleaved2MaskToContiguous(op, sourceParts, resultTypes, rewriter);
        if (failed(results)) {
            return failure();
        }
        return std::optional<SmallVector<Value>>(std::move(*results));
    }
    FailureOr<SmallVector<Value>> results =
        materializeContiguousToDeinterleaved2Mask(op, sourceParts, resultTypes, rewriter);
    if (failed(results)) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation* op, Type resultType, StringRef diagnostic, PatternRewriter& rewriter);

static FailureOr<SmallVector<Value>> materializeMaskLaneStrideUnpack(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, int64_t laneStride, PatternRewriter& rewriter)
{
    bool resultExceedsSource =
        static_cast<int64_t>(resultTypes.size()) > static_cast<int64_t>(sourceParts.size()) * laneStride;
    if (resultExceedsSource) {
        return rewriter.notifyMatchFailure(
            op, "dense mask lane_stride unpack materialization result arity "
                "does not fit source arity");
    }
    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    StringAttr lower = rewriter.getStringAttr("LOWER");
    StringAttr higher = rewriter.getStringAttr("HIGHER");
    for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
        FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
            op, resultType, "dense mask lane_stride unpack requires mask result type", rewriter);
        if (failed(maskType)) {
            return failure();
        }
        int64_t safeLaneStride = laneStride > 0 ? laneStride : 1;
        int64_t sourceIndex = resultIndex / safeLaneStride;
        int64_t part = resultIndex % safeLaneStride;
        Value source = sourceParts[sourceIndex];
        StringAttr firstPart = laneStride == 4 ? (part >= 2 ? higher : lower) : (part == 1 ? higher : lower);
        // A lane_stride=4 source is unpacked twice: the first unpack splits the
        // register into its two b16 halves, and only the second one produces the
        // result mask type.
        MaskType firstResultType =
            laneStride == kVMIDataLayoutFactor4 ? MaskType::get(op->getContext(), "b16") : *maskType;
        Value current = rewriter.create<PunpackOp>(op->getLoc(), firstResultType, source, firstPart).getResult();
        if (laneStride == kVMIDataLayoutFactor4) {
            current = rewriter.create<PunpackOp>(
                op->getLoc(), *maskType, current, part % kVMIDataLayoutPairSize == 0 ? lower : higher);
        }
        results.push_back(current);
    }
    return results;
}

static FailureOr<MaskType> getMaskLaneStrideResultType(
    Operation* op, Type resultType, StringRef diagnostic, PatternRewriter& rewriter)
{
    auto maskType = dyn_cast<MaskType>(resultType);
    if (!maskType) {
        (void)rewriter.notifyMatchFailure(op, diagnostic);
        return failure();
    }
    return maskType;
}

struct MaskLaneStridePackContext {
    Operation* op;
    PatternRewriter& rewriter;
    StringAttr lower;
    StringAttr higher;
    Value allTrue;

    FailureOr<Value> merge(Value lhs, Value rhs)
    {
        if (!allTrue) {
            FailureOr<Value> mask = createAllTrueMask(op->getLoc(), cast<MaskType>(lhs.getType()), rewriter);
            if (failed(mask)) {
                return failure();
            }
            allTrue = *mask;
        }
        return rewriter.create<PorOp>(op->getLoc(), lhs.getType(), lhs, rhs, allTrue).getResult();
    }

    FailureOr<Value> packPair(Value lowSource, std::optional<Value> highSource, MaskType pairType)
    {
        Value packed = rewriter.create<PpackOp>(op->getLoc(), pairType, lowSource, lower);
        if (!highSource) {
            return packed;
        }
        Value higherPacked = rewriter.create<PpackOp>(op->getLoc(), pairType, *highSource, higher);
        return merge(packed, higherPacked);
    }
};

static FailureOr<Value> materializeMaskLaneStridePackChunk(
    Operation* op, ValueRange sourceParts, size_t base, int64_t laneStride, MaskType maskType,
    MaskLaneStridePackContext& context, PatternRewriter& rewriter)
{
    std::optional<Value> source1;
    if (base + 1 < sourceParts.size()) {
        source1 = sourceParts[base + 1];
    }
    // The pair packs of a lane_stride=4 result land in a b16 half; only the outer
    // packs in this helper produce the result mask type.
    MaskType pairType = laneStride == kVMIDataLayoutFactor4 ? MaskType::get(op->getContext(), "b16") : maskType;
    FailureOr<Value> lowHalf = context.packPair(sourceParts[base], source1, pairType);
    if (failed(lowHalf)) {
        return failure();
    }
    Value current = *lowHalf;
    if (laneStride != kVMIDataLayoutFactor4) {
        return current;
    }
    current = rewriter.create<PpackOp>(op->getLoc(), maskType, current, context.lower);
    if (base + kVMIDataLayoutFactor2 >= sourceParts.size()) {
        return current;
    }
    std::optional<Value> source3;
    if (base + kVMIDataLayoutFourthPartIndex < sourceParts.size()) {
        source3 = sourceParts[base + kVMIDataLayoutFourthPartIndex];
    }
    FailureOr<Value> highHalf = context.packPair(sourceParts[base + kVMIDataLayoutFactor2], source3, pairType);
    if (failed(highHalf)) {
        return failure();
    }
    Value higherPacked = rewriter.create<PpackOp>(op->getLoc(), maskType, *highHalf, context.higher);
    return context.merge(current, higherPacked);
}

static FailureOr<SmallVector<Value>> materializeMaskLaneStridePack(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, int64_t laneStride, PatternRewriter& rewriter)
{
    if (sourceParts.empty()) {
        return rewriter.notifyMatchFailure(op, "dense mask lane_stride pack materialization requires source parts");
    }
    bool sourceExceedsResult =
        static_cast<int64_t>(sourceParts.size()) > static_cast<int64_t>(resultTypes.size()) * laneStride;
    if (sourceExceedsResult) {
        return rewriter.notifyMatchFailure(
            op, "dense mask lane_stride pack materialization source arity does "
                "not fit result arity");
    }
    MaskLaneStridePackContext context{
        op, rewriter, rewriter.getStringAttr("LOWER"), rewriter.getStringAttr("HIGHER"), Value()};

    SmallVector<Value> results;
    results.reserve(resultTypes.size());
    for (auto [resultIndex, resultType] : llvm::enumerate(resultTypes)) {
        FailureOr<MaskType> maskType = getMaskLaneStrideResultType(
            op, resultType, "dense mask lane_stride pack requires mask result type", rewriter);
        if (failed(maskType)) {
            return failure();
        }
        size_t base = resultIndex * static_cast<size_t>(laneStride);
        if (base >= sourceParts.size()) {
            break;
        }
        FailureOr<Value> current =
            materializeMaskLaneStridePackChunk(op, sourceParts, base, laneStride, *maskType, context, rewriter);
        if (failed(current)) {
            return failure();
        }
        results.push_back(*current);
    }
    bool resultArityMismatch = results.size() != resultTypes.size();
    if (resultArityMismatch) {
        return rewriter.notifyMatchFailure(op, "dense mask lane_stride pack materialization result arity mismatch");
    }
    return results;
}

struct MaskLaneStrideLayoutPlan {
    bool unpack;
    int64_t laneStride;
};

static std::optional<MaskLaneStrideLayoutPlan> getMaskLaneStrideLayoutPlan(
    VMILayoutAttr sourceLayout, VMILayoutAttr resultLayout)
{
    bool unpack = sourceLayout && sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 && resultLayout &&
                  resultLayout.isContiguous() && resultLayout.getLaneStride() != 1;
    bool pack = sourceLayout && sourceLayout.isContiguous() && sourceLayout.getLaneStride() != 1 && resultLayout &&
                resultLayout.isContiguous() && resultLayout.getLaneStride() == 1;
    if (!unpack && !pack) {
        return std::nullopt;
    }
    return MaskLaneStrideLayoutPlan{unpack, unpack ? resultLayout.getLaneStride() : sourceLayout.getLaneStride()};
}

static LogicalResult checkMaskLaneStrideFactor(
    Operation* op, const MaskLaneStrideLayoutPlan& plan, PatternRewriter& rewriter)
{
    bool supportedFactor = plan.laneStride == 2 || plan.laneStride == 4;
    if (supportedFactor) {
        return success();
    }
    return rewriter.notifyMatchFailure(
        op, plan.unpack ? "unsupported dense mask lane_stride unpack factor" :
                          "unsupported dense mask lane_stride pack factor");
}

FailureOr<std::optional<SmallVector<Value>>> materializeMaskLaneStrideLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    std::optional<MaskLaneStrideLayoutPlan> plan = getMaskLaneStrideLayoutPlan(sourceLayout, resultLayout);
    if (!plan) {
        return std::optional<SmallVector<Value>>{};
    }

    if (failed(checkMaskLaneStrideFactor(op, *plan, rewriter))) {
        return failure();
    }

    if (plan->unpack) {
        FailureOr<SmallVector<Value>> results =
            materializeMaskLaneStrideUnpack(op, sourceParts, resultTypes, plan->laneStride, rewriter);
        if (failed(results)) {
            return failure();
        }
        return std::optional<SmallVector<Value>>(std::move(*results));
    }
    FailureOr<SmallVector<Value>> results =
        materializeMaskLaneStridePack(op, sourceParts, resultTypes, plan->laneStride, rewriter);
    if (failed(results)) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
}

static FailureOr<std::optional<SmallVector<Value>>> materializeIdentityMaskLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    if (sourceLayout != resultLayout) {
        return std::optional<SmallVector<Value>>{};
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes, rewriter))) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(SmallVector<Value>(sourceParts.begin(), sourceParts.end()));
}

struct MaskLayoutMaterializationContext {
    Operation* op;
    ValueRange sourceParts;
    TypeRange resultTypes;
    VMILayoutAttr sourceLayout;
    VMILayoutAttr resultLayout;
    PatternRewriter& rewriter;
};

using MaskLayoutMaterializationResult = FailureOr<std::optional<SmallVector<Value>>>;

static bool didHandleMaskLayoutMaterialization(const MaskLayoutMaterializationResult& result)
{
    return failed(result) || result->has_value();
}

static bool isElementDeinterleavedMaskLayout(VMILayoutAttr layout, int64_t factor)
{
    return layout && layout.isDeinterleaved() && layout.getFactor() == factor && layout.getLaneStride() == 1;
}

static MaskLayoutMaterializationResult materializeDeinterleaved4MaskLayout(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    bool contiguousToDeint4 = sourceLayout.isContiguous() && sourceLayout.getLaneStride() == 1 &&
                              isElementDeinterleavedMaskLayout(resultLayout, 4);
    bool deint4ToContiguous = isElementDeinterleavedMaskLayout(sourceLayout, 4) && resultLayout.isContiguous() &&
                              resultLayout.getLaneStride() == 1;
    if (!contiguousToDeint4 && !deint4ToContiguous) {
        return std::optional<SmallVector<Value>>{};
    }
    // As for factor 2: an unequal part count is the grouped staging case.
    if (sourceParts.size() != resultTypes.size()) {
        if (contiguousToDeint4) {
            return materializeStagingContiguousToDeintMaskLayout(
                op, sourceParts, resultTypes, kVMIDataLayoutFactor4, rewriter);
        }
        return materializeStagingDeintToContiguousMaskLayout(
            op, sourceParts, resultTypes, kVMIDataLayoutFactor4, rewriter);
    }
    bool invalidArity = sourceParts.empty() || resultTypes.size() % 4 != 0;
    if (invalidArity) {
        (void)rewriter.notifyMatchFailure(op, "deinterleaved=4 mask layout materialization requires 4*N parts");
        return failure();
    }
    if (failed(verifyIdentityPartForwarding(op, sourceParts, resultTypes, rewriter))) {
        return failure();
    }
    FailureOr<SmallVector<Value>> results = contiguousToDeint4 ?
                                                materializeStagingContiguousToDeintMaskLayout(
                                                    op, sourceParts, resultTypes, kVMIDataLayoutFactor4, rewriter) :
                                                materializeStagingDeintToContiguousMaskLayout(
                                                    op, sourceParts, resultTypes, kVMIDataLayoutFactor4, rewriter);
    if (failed(results)) {
        return failure();
    }
    return std::optional<SmallVector<Value>>(std::move(*results));
}

// Block permutations are different from element deinterleaving. Encode each
// predicate as a 0/1 word, select its logical lanes, then rebuild the predicate.
struct BlockMaskPermutation {
    Operation* op;
    VRegType wordType;
    MaskType maskType;
    Value allMask;
    int64_t factor;
    bool toBlock;
    PatternRewriter& rewriter;
};

static Value buildBlockMaskRemainder(const BlockMaskPermutation& context, Value value, int64_t shift)
{
    auto& rewriter = context.rewriter;
    Location loc = context.op->getLoc();
    Value amount = createI16Constant(loc, shift, rewriter);
    Value quotient = rewriter.create<VshrsOp>(loc, context.wordType, value, amount, context.allMask);
    Value base = rewriter.create<VshlsOp>(loc, context.wordType, quotient, amount, context.allMask);
    return rewriter.create<VsubOp>(loc, context.wordType, value, base, context.allMask);
}

static Value buildBlockMaskLogicalLanes(const BlockMaskPermutation& context, int64_t part)
{
    auto& rewriter = context.rewriter;
    Location loc = context.op->getLoc();
    Value base = createI32Constant(loc, context.toBlock ? 0 : part * 64, rewriter);
    Value lanes = rewriter.create<VciOp>(loc, context.wordType, base, StringAttr{});
    if (!context.toBlock) {
        return lanes;
    }
    Value three = createI16Constant(loc, 3, rewriter);
    Value shift = createI16Constant(loc, context.factor == 2 ? 4 : 5, rewriter);
    Value blocks = rewriter.create<VshrsOp>(loc, context.wordType, lanes, three, context.allMask);
    Value blockBase = rewriter.create<VshlsOp>(loc, context.wordType, blocks, shift, context.allMask);
    Value inBlock = buildBlockMaskRemainder(context, lanes, 3);
    Value logical = rewriter.create<VaddOp>(loc, context.wordType, blockBase, inBlock, context.allMask);
    if (part == 0) {
        return logical;
    }
    Value delta = createI32Constant(loc, part * 8, rewriter);
    return rewriter.create<VaddsOp>(loc, context.wordType, logical, delta, context.allMask);
}

static std::pair<Value, Value> buildBlockMaskSourceIndices(const BlockMaskPermutation& context, Value logical)
{
    auto& rewriter = context.rewriter;
    Location loc = context.op->getLoc();
    if (context.toBlock) {
        Value six = createI16Constant(loc, 6, rewriter);
        Value part = rewriter.create<VshrsOp>(loc, context.wordType, logical, six, context.allMask);
        Value lane = buildBlockMaskRemainder(context, logical, 6);
        return {part, lane};
    }
    Value three = createI16Constant(loc, 3, rewriter);
    Value shift = createI16Constant(loc, context.factor == 2 ? 4 : 5, rewriter);
    Value blocks = rewriter.create<VshrsOp>(loc, context.wordType, logical, three, context.allMask);
    Value part = buildBlockMaskRemainder(context, blocks, context.factor == 2 ? 1 : 2);
    Value rows = rewriter.create<VshrsOp>(loc, context.wordType, logical, shift, context.allMask);
    Value rowBase = rewriter.create<VshlsOp>(loc, context.wordType, rows, three, context.allMask);
    Value inBlock = buildBlockMaskRemainder(context, logical, 3);
    Value lane = rewriter.create<VaddOp>(loc, context.wordType, rowBase, inBlock, context.allMask);
    return {part, lane};
}

static Value materializeBlockMaskPart(
    const BlockMaskPermutation& context, ValueRange encodedParts, int64_t resultPart, Value zero)
{
    auto& rewriter = context.rewriter;
    Location loc = context.op->getLoc();
    Value logical = buildBlockMaskLogicalLanes(context, resultPart);
    auto [sourcePart, sourceLane] = buildBlockMaskSourceIndices(context, logical);
    Value result;
    for (auto [index, source] : llvm::enumerate(encodedParts)) {
        Value selected = rewriter.create<VselrOp>(loc, context.wordType, source, sourceLane);
        // Every live output lane belongs to exactly one source carrier.
        if (!result) {
            result = selected;
            continue;
        }
        Value scalar = createI32Constant(loc, static_cast<int64_t>(index), rewriter);
        Value membership = rewriter.create<VcmpsOp>(
            loc, context.maskType, sourcePart, scalar, context.allMask, rewriter.getStringAttr("eq"));
        result = rewriter.create<VselOp>(loc, context.wordType, selected, result, membership);
    }
    return rewriter.create<VcmpsOp>(loc, context.maskType, result, zero, context.allMask, rewriter.getStringAttr("ne"));
}

static MaskLayoutMaterializationResult materializeBlockMaskLayout(const MaskLayoutMaterializationContext& input)
{
    bool toBlock = input.sourceLayout.isContiguous() && input.sourceLayout.getLaneStride() == 1 &&
                   input.resultLayout.isBlockDeinterleaved();
    bool toContiguous = input.sourceLayout.isBlockDeinterleaved() && input.resultLayout.isContiguous() &&
                        input.resultLayout.getLaneStride() == 1;
    if (!toBlock && !toContiguous) {
        return std::optional<SmallVector<Value>>{};
    }
    int64_t factor = toBlock ? input.resultLayout.getFactor() : input.sourceLayout.getFactor();
    auto maskType = MaskType::get(input.rewriter.getContext(), "b32");
    bool invalidShape = (factor != 2 && factor != 4) || input.sourceParts.size() != static_cast<size_t>(factor) ||
                        input.resultTypes.size() != input.sourceParts.size() ||
                        llvm::any_of(input.sourceParts, [maskType](Value v) { return v.getType() != maskType; });
    if (invalidShape) {
        return failure();
    }
    auto& rewriter = input.rewriter;
    Location loc = input.op->getLoc();
    FailureOr<Value> allMask = createAllTrueMask(loc, maskType, rewriter);
    if (failed(allMask)) {
        return failure();
    }
    auto wordType = VRegType::get(rewriter.getContext(), 64, rewriter.getI32Type());
    Value zero = createI32Constant(loc, 0, rewriter);
    Value one = createI32Constant(loc, 1, rewriter);
    Value zeroVector = rewriter.create<VdupOp>(loc, wordType, zero, *allMask, StringAttr{});
    Value oneVector = rewriter.create<VdupOp>(loc, wordType, one, *allMask, StringAttr{});
    SmallVector<Value> encoded;
    for (Value mask : input.sourceParts) {
        encoded.push_back(rewriter.create<VselOp>(loc, wordType, oneVector, zeroVector, mask));
    }
    BlockMaskPermutation context{input.op, wordType, maskType, *allMask, factor, toBlock, rewriter};
    SmallVector<Value> results;
    for (int64_t part = 0; part < factor; ++part) {
        results.push_back(materializeBlockMaskPart(context, encoded, part, zero));
    }
    return std::optional<SmallVector<Value>>(std::move(results));
}

static MaskLayoutMaterializationResult tryMaskLayoutMaterializers(const MaskLayoutMaterializationContext& context)
{
    MaskLayoutMaterializationResult identity = materializeIdentityMaskLayout(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.rewriter);
    if (didHandleMaskLayoutMaterialization(identity)) {
        return identity;
    }

    MaskLayoutMaterializationResult block = materializeBlockMaskLayout(context);
    if (didHandleMaskLayoutMaterialization(block)) {
        return block;
    }

    MaskLayoutMaterializationResult deinterleaved2 = materializeDeinterleaved2MaskLayout(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.rewriter);
    if (didHandleMaskLayoutMaterialization(deinterleaved2)) {
        return deinterleaved2;
    }

    MaskLayoutMaterializationResult deinterleaved4 = materializeDeinterleaved4MaskLayout(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.rewriter);
    if (didHandleMaskLayoutMaterialization(deinterleaved4)) {
        return deinterleaved4;
    }

    return materializeMaskLaneStrideLayout(
        context.op, context.sourceParts, context.resultTypes, context.sourceLayout, context.resultLayout,
        context.rewriter);
}

FailureOr<SmallVector<Value>> materializeMaskLayoutConversion(
    Operation* op, ValueRange sourceParts, TypeRange resultTypes, VMILayoutAttr sourceLayout,
    VMILayoutAttr resultLayout, PatternRewriter& rewriter)
{
    if (!sourceLayout || !resultLayout) {
        (void)rewriter.notifyMatchFailure(
            op, "mask layout materialization requires assigned source/result "
                "layouts");
        return failure();
    }

    MaskLayoutMaterializationContext context{op, sourceParts, resultTypes, sourceLayout, resultLayout, rewriter};
    FailureOr<std::optional<SmallVector<Value>>> results = tryMaskLayoutMaterializers(context);
    if (failed(results)) {
        return failure();
    }
    if (results->has_value()) {
        return std::move(**results);
    }

    (void)rewriter.notifyMatchFailure(op, "unsupported VMI mask layout materialization");
    return failure();
}

int getMaskGranularityRank(StringRef granularity)
{
    if (granularity == "b8") {
        return 0;
    }
    if (granularity == "b16") {
        return 1;
    }
    if (granularity == "b32") {
        return kVMIDataLayoutMaskGranularityRankB32;
    }
    return -1;
}

StringRef getMaskGranularityForRank(int rank)
{
    switch (rank) {
        case 0:
            return "b8";
        case 1:
            return "b16";
        case kVMIDataLayoutMaskGranularityRankB32:
            return "b32";
        default:
            return "";
    }
}

LogicalResult checkSupportedMaskGranularityMaterialization(
    VMIMaskType sourceType, VMIMaskType resultType, std::string* reason)
{
    auto fail = [&reason](const Twine& message) -> LogicalResult {
        if (reason) {
            *reason = message.str();
        }
        return failure();
    };

    bool laneCountMismatch = sourceType.getElementCount() != resultType.getElementCount();
    if (laneCountMismatch) {
        return fail("requires source and result mask lane counts to match");
    }
    bool layoutMismatch = sourceType.getLayoutAttr() != resultType.getLayoutAttr();
    if (layoutMismatch) {
        return fail("requires source and result mask layouts to match");
    }

    bool nonConcreteGranularity = !VMIMaskType::isConcreteGranularity(sourceType.getGranularity()) ||
                                  !VMIMaskType::isConcreteGranularity(resultType.getGranularity());
    if (nonConcreteGranularity) {
        return fail("requires concrete b8/b16/b32 source and result "
                    "granularities");
    }

    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
    bool missingArity = failed(sourceArity) || failed(resultArity);
    if (missingArity) {
        return fail("requires computable source/result physical arity");
    }
    if (*sourceArity < 1 || *resultArity < 1) {
        return fail("requires non-empty source/result physical arity");
    }

    return success();
}

FailureOr<SmallVector<Value>> materializeWideningMaskGranularityPart(
    Operation* op, MaskType resultMaskType, ValueRange sourceParts, int64_t sourceOffset, int64_t sourceChunks,
    int64_t resultChunks, PatternRewriter& rewriter)
{
    SmallVector<Value> results;
    auto partAttr = StringAttr::get(op->getContext(), "LOWER");
    auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
    int64_t produced = 0;
    for (int64_t chunk = 0; chunk < sourceChunks && produced < resultChunks; ++chunk) {
        Value source = sourceParts[sourceOffset + chunk];
        results.push_back(rewriter.create<PunpackOp>(op->getLoc(), resultMaskType, source, partAttr).getResult());
        ++produced;
        if (produced < resultChunks) {
            results.push_back(rewriter.create<PunpackOp>(op->getLoc(), resultMaskType, source, higherAttr).getResult());
            ++produced;
        }
    }
    if (produced != resultChunks) {
        (void)rewriter.notifyMatchFailure(
            op, "widening mask granularity conversion produced the wrong number "
                "of result chunks");
        return failure();
    }
    return results;
}

static FailureOr<Value> materializeNarrowingMaskChunk(
    Operation* op, MaskType resultMaskType, ValueRange sourceParts, int64_t sourceOffset, int64_t sourceChunks,
    int64_t& consumed, Value& allTrue, StringAttr lowerAttr, StringAttr higherAttr, PatternRewriter& rewriter)
{
    if (consumed >= sourceChunks) {
        (void)rewriter.notifyMatchFailure(op, "narrowing mask granularity conversion ran out of source chunks");
        return failure();
    }
    Value lowerSource = sourceParts[sourceOffset + consumed++];
    Value packed = rewriter.create<PpackOp>(op->getLoc(), resultMaskType, lowerSource, lowerAttr).getResult();
    if (consumed >= sourceChunks) {
        return packed;
    }
    Value higherSource = sourceParts[sourceOffset + consumed++];
    Value higher = rewriter.create<PpackOp>(op->getLoc(), resultMaskType, higherSource, higherAttr).getResult();
    if (!allTrue) {
        FailureOr<Value> mask = createAllTrueMask(op->getLoc(), resultMaskType, rewriter);
        if (failed(mask)) {
            (void)rewriter.notifyMatchFailure(op, "failed to create all-true mask for ppack merge");
            return failure();
        }
        allTrue = *mask;
    }
    return rewriter.create<PorOp>(op->getLoc(), resultMaskType, packed, higher, allTrue).getResult();
}

FailureOr<SmallVector<Value>> materializeNarrowingMaskGranularityPart(
    Operation* op, MaskType resultMaskType, ValueRange sourceParts, int64_t sourceOffset, int64_t sourceChunks,
    int64_t resultChunks, PatternRewriter& rewriter)
{
    auto fail = [&op, &rewriter](const Twine& message) -> FailureOr<SmallVector<Value>> {
        (void)rewriter.notifyMatchFailure(op, message);
        return failure();
    };
    auto lowerAttr = StringAttr::get(op->getContext(), "LOWER");
    auto higherAttr = StringAttr::get(op->getContext(), "HIGHER");
    SmallVector<Value> results;
    Value allTrue;
    int64_t consumed = 0;
    for (int64_t chunk = 0; chunk < resultChunks; ++chunk) {
        FailureOr<Value> packed = materializeNarrowingMaskChunk(
            op, resultMaskType, sourceParts, sourceOffset, sourceChunks, consumed, allTrue, lowerAttr, higherAttr,
            rewriter);
        if (failed(packed)) {
            return failure();
        }
        results.push_back(*packed);
    }
    if (consumed != sourceChunks) {
        return fail("narrowing mask granularity conversion left unused source "
                    "chunks");
    }
    return results;
}

static FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityPart(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts, int sourceRank,
    int resultRank, MaskType resultMaskType, int64_t part, int64_t sourceOffset, PatternRewriter& rewriter)
{
    FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
    FailureOr<int64_t> resultChunks = getVMITypeChunksInPart(resultType, part);
    bool invalidChunkCounts = failed(sourceChunks) || failed(resultChunks);
    if (invalidChunkCounts) {
        (void)rewriter.notifyMatchFailure(op, "requires computable source/result chunks per layout part");
        return failure();
    }
    bool widening = resultRank > sourceRank;
    if (widening) {
        return materializeWideningMaskGranularityPart(
            op, resultMaskType, sourceParts, sourceOffset, *sourceChunks, *resultChunks, rewriter);
    }
    return materializeNarrowingMaskGranularityPart(
        op, resultMaskType, sourceParts, sourceOffset, *sourceChunks, *resultChunks, rewriter);
}

struct MaskGranularityConversionPlan {
    int sourceRank;
    int resultRank;
    int64_t sourceArity;
    int64_t layoutFactor;
    MaskType resultMaskType;
};

static FailureOr<MaskGranularityConversionPlan> buildMaskGranularityConversionPlan(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts, PatternRewriter& rewriter)
{
    auto fail = [&op, &rewriter](const Twine& message) -> FailureOr<MaskGranularityConversionPlan> {
        (void)rewriter.notifyMatchFailure(op, message);
        return failure();
    };
    int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
    int resultRank = getMaskGranularityRank(resultType.getGranularity());
    bool nonAdjacent = std::abs(sourceRank - resultRank) != 1;
    if (nonAdjacent) {
        return fail("mask granularity conversion must be adjacent");
    }
    FailureOr<int64_t> sourceArity = getVMIPhysicalArity(sourceType);
    FailureOr<int64_t> factor = getVMITypeLayoutFactor(sourceType);
    bool sourceArityMismatch =
        failed(sourceArity) || failed(factor) || static_cast<int64_t>(sourceParts.size()) != *sourceArity;
    if (sourceArityMismatch) {
        return fail("source mask part count does not match source VMI type");
    }
    return MaskGranularityConversionPlan{
        sourceRank, resultRank, *sourceArity, *factor, MaskType::get(op->getContext(), resultType.getGranularity())};
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityParts(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts,
    const MaskGranularityConversionPlan& plan, PatternRewriter& rewriter)
{
    // A group-slot layout indexes its physical carriers by group slot, not by
    // dense lane chunk, so the chunked path below cannot express a multi-carrier
    // conversion: its part count comes from the dense layout factor, which is 1
    // for group slots, and it therefore returns far fewer parts than the physical
    // arity. Equal-arity group-slot masks convert one carrier at a time instead.
    // A single-carrier conversion (arity 1) keeps the chunked path, which already
    // folds it for equal layouts.
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
    bool perCarrierGroupSlots = sourceType.getLayoutAttr().isGroupSlots() &&
                                resultType.getLayoutAttr().isGroupSlots() && succeeded(resultArity) &&
                                plan.sourceArity == *resultArity && plan.sourceArity > 1;
    if (perCarrierGroupSlots) {
        auto partAttr = StringAttr::get(op->getContext(), "LOWER");
        bool widening = plan.resultRank > plan.sourceRank;
        SmallVector<Value> carriers;
        carriers.reserve(sourceParts.size());
        for (Value source : sourceParts) {
            if (widening) {
                carriers.push_back(
                    rewriter.create<PunpackOp>(op->getLoc(), plan.resultMaskType, source, partAttr).getResult());
            } else {
                carriers.push_back(
                    rewriter.create<PpackOp>(op->getLoc(), plan.resultMaskType, source, partAttr).getResult());
            }
        }
        return carriers;
    }

    SmallVector<Value> results;
    int64_t sourceOffset = 0;
    for (int64_t part = 0; part < plan.layoutFactor; ++part) {
        FailureOr<int64_t> sourceChunks = getVMITypeChunksInPart(sourceType, part);
        if (failed(sourceChunks)) {
            (void)rewriter.notifyMatchFailure(op, "requires computable source chunks per layout part");
            return failure();
        }
        FailureOr<SmallVector<Value>> partResults = materializeAdjacentMaskGranularityPart(
            op, sourceType, resultType, sourceParts, plan.sourceRank, plan.resultRank, plan.resultMaskType, part,
            sourceOffset, rewriter);
        if (failed(partResults)) {
            return failure();
        }
        results.append(*partResults);
        sourceOffset += *sourceChunks;
    }
    return results;
}

static LogicalResult checkMaskGranularityResultArity(
    Operation* op, VMIMaskType resultType, size_t resultCount, PatternRewriter& rewriter)
{
    FailureOr<int64_t> resultArity = getVMIPhysicalArity(resultType);
    bool resultArityMismatch = failed(resultArity) || static_cast<int64_t>(resultCount) != *resultArity;
    if (resultArityMismatch) {
        (void)rewriter.notifyMatchFailure(op, "mask granularity conversion result count mismatch");
        return failure();
    }
    return success();
}

FailureOr<SmallVector<Value>> materializeAdjacentMaskGranularityConversion(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts, PatternRewriter& rewriter)
{
    FailureOr<MaskGranularityConversionPlan> plan =
        buildMaskGranularityConversionPlan(op, sourceType, resultType, sourceParts, rewriter);
    if (failed(plan)) {
        return failure();
    }
    FailureOr<SmallVector<Value>> results =
        materializeMaskGranularityParts(op, sourceType, resultType, sourceParts, *plan, rewriter);
    if (failed(results)) {
        return failure();
    }

    if (failed(checkMaskGranularityResultArity(op, resultType, results->size(), rewriter))) {
        return failure();
    }
    return *results;
}

static FailureOr<SmallVector<Value>> materializeMaskGranularityStep(
    Operation* op, VMIMaskType currentType, StringRef nextGranularity, ValueRange currentParts,
    PatternRewriter& rewriter)
{
    VMIMaskType nextType =
        VMIMaskType::get(op->getContext(), currentType.getElementCount(), nextGranularity, currentType.getLayoutAttr());
    return materializeAdjacentMaskGranularityConversion(op, currentType, nextType, currentParts, rewriter);
}

static FailureOr<VMIMaskType> buildNextMaskGranularityType(
    VMIMaskType currentType, int rank, PatternRewriter& rewriter, Operation* op)
{
    StringRef granularity = getMaskGranularityForRank(rank);
    if (granularity.empty()) {
        (void)rewriter.notifyMatchFailure(op, "invalid target mask granularity rank");
        return failure();
    }
    return VMIMaskType::get(op->getContext(), currentType.getElementCount(), granularity, currentType.getLayoutAttr());
}

static FailureOr<SmallVector<Value>> materializeMaskGranularitySteps(
    Operation* op, VMIMaskType sourceType, ValueRange sourceParts, int sourceRank, int resultRank,
    PatternRewriter& rewriter)
{
    VMIMaskType currentType = sourceType;
    SmallVector<Value> currentParts(sourceParts.begin(), sourceParts.end());
    int currentRank = sourceRank;
    while (currentRank != resultRank) {
        bool ascending = currentRank < resultRank;
        currentRank += ascending ? 1 : -1;
        FailureOr<VMIMaskType> nextType = buildNextMaskGranularityType(currentType, currentRank, rewriter, op);
        if (failed(nextType)) {
            return failure();
        }
        FailureOr<SmallVector<Value>> nextParts =
            materializeMaskGranularityStep(op, currentType, nextType->getGranularity(), currentParts, rewriter);
        if (failed(nextParts)) {
            return failure();
        }
        currentType = *nextType;
        currentParts = std::move(*nextParts);
    }
    return currentParts;
}

struct MaskGranularityRoute {
    int sourceRank;
    int resultRank;
    bool adjacent;
};

static FailureOr<MaskGranularityRoute> classifyMaskGranularityRoute(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, PatternRewriter& rewriter)
{
    auto fail = [&rewriter, op](const Twine& message) -> FailureOr<MaskGranularityRoute> {
        (void)rewriter.notifyMatchFailure(op, message);
        return failure();
    };
    int sourceRank = getMaskGranularityRank(sourceType.getGranularity());
    int resultRank = getMaskGranularityRank(resultType.getGranularity());
    if (sourceRank < 0 || resultRank < 0) {
        return fail("requires concrete source and result mask granularity ranks");
    }
    return MaskGranularityRoute{sourceRank, resultRank, std::abs(sourceRank - resultRank) == 1};
}

FailureOr<SmallVector<Value>> materializeMaskGranularityConversion(
    Operation* op, VMIMaskType sourceType, VMIMaskType resultType, ValueRange sourceParts, PatternRewriter& rewriter)
{
    std::string reason;
    if (failed(checkSupportedMaskGranularityMaterialization(sourceType, resultType, &reason))) {
        (void)rewriter.notifyMatchFailure(op, reason);
        return failure();
    }

    FailureOr<MaskGranularityRoute> route = classifyMaskGranularityRoute(op, sourceType, resultType, rewriter);
    if (failed(route)) {
        return failure();
    }
    if (route->adjacent) {
        return materializeAdjacentMaskGranularityConversion(op, sourceType, resultType, sourceParts, rewriter);
    }

    return materializeMaskGranularitySteps(op, sourceType, sourceParts, route->sourceRank, route->resultRank, rewriter);
}

static SmallVector<Type> repeatMaskPartType(Type partType, int64_t arity)
{
    SmallVector<Type> types;
    types.reserve(arity);
    for (int64_t i = 0; i < arity; ++i) {
        types.push_back(partType);
    }
    return types;
}

FailureOr<SmallVector<Type>> getConvertedMaskPartTypes(VMIMaskType type)
{
    FailureOr<int64_t> arity = getVMIPhysicalArity(type);
    FailureOr<StringRef> physicalGranularity = getVMIMaskPhysicalGranularity(type);
    bool invalidMaskTypes = failed(arity) || failed(physicalGranularity) || *arity < 0;
    if (invalidMaskTypes) {
        return failure();
    }
    Type partType = MaskType::get(type.getContext(), *physicalGranularity);
    return repeatMaskPartType(partType, *arity);
}

static FailureOr<VMILayoutAttr> getVMIMaskPhysicalCarrierLayout(VMIMaskType type)
{
    auto layout = getVMIMaskCarrierLayout(type.getLayoutAttr());
    return layout ? FailureOr<VMILayoutAttr>(layout) : FailureOr<VMILayoutAttr>(failure());
}

static FailureOr<VMIMaskType> getVMIMaskPhysicalCarrierType(VMIMaskType type)
{
    FailureOr<StringRef> physicalGranularity = getVMIMaskPhysicalGranularity(type);
    FailureOr<VMILayoutAttr> physicalLayout = getVMIMaskPhysicalCarrierLayout(type);
    bool missingPhysicalCarrier = failed(physicalGranularity) || failed(physicalLayout);
    if (missingPhysicalCarrier) {
        return failure();
    }
    return VMIMaskType::get(type.getContext(), type.getElementCount(), *physicalGranularity, *physicalLayout);
}

static bool isElementDeinterleavedLayout(VMILayoutAttr layout, int64_t factor)
{
    return layout && layout.isDeinterleaved() && layout.getFactor() == factor && layout.getLaneStride() == 1;
}

FailureOr<Value> createAllFalseMaskLike(Location loc, Value value, PatternRewriter& rewriter)
{
    auto maskType = dyn_cast<MaskType>(value.getType());
    if (!maskType) {
        return failure();
    }
    return createPrefixMask(loc, maskType, "PAT_ALLF", rewriter);
}

FailureOr<std::array<Value, kVMIDataLayoutFactor4>> materializeFactor4DeintToContiguousGroup(
    Operation* op, ArrayRef<Value> sources, TypeRange resultTypes, size_t resultOffset, PatternRewriter& rewriter)
{
    auto fail = [&op, &rewriter](const Twine& message) -> FailureOr<std::array<Value, kVMIDataLayoutFactor4>> {
        (void)rewriter.notifyMatchFailure(op, message);
        return failure();
    };
    bool invalidSources = sources.size() != 4;
    bool missingResultTypes = resultTypes.empty();
    if (invalidSources || missingResultTypes) {
        return fail("factor-4 staging mask conversion requires four sources");
    }
    auto resultTypeAt = [resultTypes, resultOffset](size_t offset) -> Type {
        size_t index = resultOffset + offset;
        return index < resultTypes.size() ? resultTypes[index] : resultTypes[resultTypes.size() - 1];
    };
    FailureOr<std::pair<Value, Value>> even =
        createPredicateIntlv(op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[0], sources[2], rewriter);
    FailureOr<std::pair<Value, Value>> odd =
        createPredicateIntlv(op->getLoc(), resultTypeAt(0), resultTypeAt(1), sources[1], sources[3], rewriter);
    if (failed(even)) {
        return fail("unsupported predicate intlv staging mask type");
    }
    if (failed(odd)) {
        return fail("unsupported predicate intlv staging mask type");
    }
    FailureOr<std::pair<Value, Value>> low =
        createPredicateIntlv(op->getLoc(), resultTypeAt(0), resultTypeAt(1), even->first, odd->first, rewriter);
    FailureOr<std::pair<Value, Value>> high =
        createPredicateIntlv(op->getLoc(), resultTypeAt(2), resultTypeAt(3), even->second, odd->second, rewriter);
    if (failed(low)) {
        return fail("unsupported predicate intlv staging mask type");
    }
    if (failed(high)) {
        return fail("unsupported predicate intlv staging mask type");
    }
    return std::array<Value, kVMIDataLayoutFactor4>{low->first, low->second, high->first, high->second};
}

static FailureOr<std::array<Value, kVMIDataLayoutPairSize>> materializeFactor2DeintToContiguousGroup(
    Operation* op, ArrayRef<Value> sources, TypeRange resultTypes, size_t resultOffset, PatternRewriter& rewriter)
{
    auto fail = [&op, &rewriter](const Twine& message) -> FailureOr<std::array<Value, kVMIDataLayoutPairSize>> {
        (void)rewriter.notifyMatchFailure(op, message);
        return failure();
    };
    bool invalidSources = sources.size() != 2 || resultTypes.empty();
    if (invalidSources) {
        return fail("factor-2 staging mask conversion requires two sources");
    }
    size_t first = std::min(resultOffset, resultTypes.size() - 1);
    size_t second = std::min(resultOffset + 1, resultTypes.size() - 1);
    FailureOr<std::pair<Value, Value>> materialized =
        createPredicateIntlv(op->getLoc(), resultTypes[first], resultTypes[second], sources[0], sources[1], rewriter);
    if (failed(materialized)) {
        return fail("unsupported predicate intlv staging mask type");
    }
    return std::array<Value, kVMIDataLayoutPairSize>{materialized->first, materialized->second};
}
