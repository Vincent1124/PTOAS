# ADR: verify frontend peer contracts at stable module boundaries

Status: proposed

Issue: https://github.com/hw-native-sys/PTOAS/issues/1640

## Context

The reported Linux aarch64 PTOAS 0.67 cores enter
`ImportReservedBufferOp::verify()` on an MLIR function-pass worker while
other workers rewrite IR. That verifier resolves a peer function and walks
its body to find a named reserve. The AIC/AIV fixpipe verifiers also walk
peer bodies. An operation pass may not inspect sibling bodies while other
passes can mutate them. Serializing only frontend pipe lowering does not
protect verifiers invoked by earlier function pipelines.

Reference: [MLIR operation pass restrictions](https://mlir.llvm.org/docs/PassManagement/#operation-pass).

The public evidence identifies this unsafe path. The stripped native stacks
and intermittent failure do not establish that this is the only possible
cause of the reported crashes.

## Decision

Keep function-local checks in operation verifiers. Move peer symbol and
reserve resolution, and the AIC/AIV cross-function fixpipe contract checks,
to `verifyFrontendPeerContracts(ModuleOp)`. Invoke it before the driver splits
backend jobs, while all child modules are present, and before pre-backend
normalization in the shared compiler entry used by embedded compilation.
Both calls run outside nested function pass managers and before IR mutation.

Keep the existing peer lookup rules, diagnostic messages, buffer matching,
fixpipe layout/quant/relu agreement, producer/consumer element types and slot
capacity checks. Ordinary MLIR parse/verify now checks local well-formedness;
full PTOAS compilation additionally checks the frontend peer contracts. A
custom pass pipeline using the dialect directly must invoke the module-level
helper at a stable boundary before lowering peer-connected frontend IR.

Do not disable MLIR threading or subprocess concurrency. A symbol-user
interface alone does not cover our fallback lookup across backend child
modules. Deferring checks until after those children have been extracted
would lose the shared symbol context.

## Validation and consequences

A deterministic regression canonicalizes a peer's dead region containing a
reserve while an import remains in a sibling function. Function verification
must complete; subsequent PTOAS compilation must reject the missing reserve.
A missing peer is likewise accepted by local verification and rejected by
full compilation. Duplicate imports remain an operation-verifier error.
Existing invalid fixpipe cases continue to diagnose through compilation; a
layout mismatch also tests the local/module verification boundary.

Replay the three supplied inputs with normal MLIR threading and bounded
process concurrency. Retain input/output hashes and failure logs. Passing
replays on macOS arm64 are local regression evidence, not acceptance of the
original Linux aarch64 wheel or hardware numerical correctness. Keep the issue
open for confirmation with a rebuilt Linux wheel containing this change.
