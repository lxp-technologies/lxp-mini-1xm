# Fix — DJL NDManager lifetime during repeated evaluation/training

We have isolated a native PyTorch/DJL crash while scaling `lxp-mini-1xm`.

Do not apply speculative workarounds such as simply reducing the validation batch count.

Investigate and fix the NDArray ownership/lifetime problem.

## Reproduction evidence

The following all succeed with the 17,308,032 parameter TinyStories configuration:

* forward B=1, T=4
* forward B=16, T=256
* backward B=1, T=64
* backward B=4, T=256
* backward B=8, T=256
* backward B=16, T=256
* AdamW update
* gradient clipping
* gradient accumulation=4 with B=16, T=256
* real `train corpus`
* validation with `maxValidationBatches=1`
* checkpointing
* one-token sample generation

However:

```text
maxValidationBatches=20
```

reliably crashes the native PyTorch runtime with:

```text
EXCEPTION_ACCESS_VIOLATION
Problematic frame:
C [c10.dll+0x85bb4]
```

using:

```text
DJL PyTorch 2.7.1
Temurin 25.0.2
Windows amd64
CPU
```

The crash occurs before the update progress line is printed.

## Primary hypothesis

Audit NDManager ownership of ALL forward-pass intermediates.

DJL's documented ownership rule is that the result of an NDArray operation is normally attached to the manager of its first NDArray operand.

Pay particular attention to `TokenEmbedding`.

Currently it performs an embedding lookup from the parameter array:

```kotlin
val weight = parameterStore.getValue(...)
return NDList(weight.get(NDIndex("{}", tokenIds)))
```

`weight` belongs to the long-lived model/parameter manager.

Determine experimentally which NDManager owns the resulting embedding tensor.

If the embedding output is attached to the model/root manager rather than the per-batch temporary manager, downstream Transformer intermediates may consequently also become attached to the root manager and survive batch disposal.

This could explain why:

```text
1 validation batch  -> succeeds
20 validation batches -> native crash
```

## Required investigation

Instrument manager ownership and resource counts.

Use appropriate DJL diagnostics such as:

* manager hierarchy inspection;
* `debugDump()`;
* manager resource counts if available;
* temporary manager naming;
* `cap()` as a diagnostic technique where appropriate.

Measure resource counts:

```text
before evaluation
after validation batch 1
after validation batch 2
...
after validation batch N
after temporary manager close
```

The number of transient NDArrays owned by the long-lived manager must NOT grow monotonically across validation batches.

Also audit training micro-batches and autoregressive generation for the same ownership issue.

## Required design

Establish a clear ownership rule:

```text
model/root manager
    └── parameters and intentionally persistent caches ONLY

batch/evaluation submanager
    └── inputs
    └── embeddings
    └── attention intermediates
    └── FFN intermediates
    └── logits
    └── loss
    └── all other transient forward tensors
```

Closing the batch/evaluation manager must release all transient tensors from that forward pass.

Do not manually close random intermediary NDArrays as a patch unless their ownership is clearly understood.

Prefer a systematic NDManager-scoping solution.

Consider DJL's supported temporary-attachment / return patterns where appropriate (`tempAttachAll`, `ret`, explicit attachment), but verify semantics against the DJL version used by this project.

## TokenEmbedding

Specifically test:

```kotlin
weight.manager
tokenIds.manager
embeddingOutput.manager
```

The resulting embedding must have batch/evaluation lifetime, not model lifetime.

Add a regression test proving this ownership property.

## DecoderLanguageModel / child blocks

Audit every operation where a model parameter is the first NDArray operand.

Examples include:

* embedding lookup;
* indexing;
* operations between parameter arrays and transient arrays;
* tied language-model head;
* any cache interaction.

Ensure a parameter-owned array does not accidentally become the ownership source for transient results.

## Evaluation regression test

Create a test that performs many repeated forwards/evaluation batches, not merely one.

For example:

```text
100+ validation batches
```

with a small model so CI remains fast.

Verify:

1. all batches finish;
2. loss remains finite;
3. gradients are not computed;
4. model parameter count remains unchanged;
5. long-lived manager resource count does not grow with batch count;
6. temporary managers close successfully.

The important assertion is not only “no exception”.

We need evidence that transient native resources are actually released.

## Real-size diagnostic

After the fix, manually verify with the 17.3M configuration:

```text
B=16
T=256
maxValidationBatches=20
```

The command that currently crashes should complete.

Then repeat enough validation batches to establish that memory consumption reaches a stable plateau rather than growing continuously.

## Documentation

Add a section to:

```text
docs/architecture/djl-memory-management.md
```

explaining:

* why Java GC is insufficient for DJL native memory;
* NDManager hierarchy;
* parameter lifetime vs batch lifetime;
* first-NDArray ownership rule;
* how an embedding lookup can accidentally attach transient tensors to model lifetime;
* how the regression test detects this class of bug.

Also create the corresponding lab note for this fix.

## Important

Do NOT fix this merely by:

* lowering `maxValidationBatches`;
* lowering batch size;
* forcing `System.gc()`;
* adding sleeps;
* increasing JVM heap;
* catching the native crash;
* switching JDK versions without evidence.

Those approaches may hide the underlying lifetime bug.

First establish the actual NDManager ownership chain and prove the fix with instrumentation.

Stop after this corrective PR.
