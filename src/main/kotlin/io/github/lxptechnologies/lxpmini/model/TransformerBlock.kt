package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList

class TransformerBlock(
    parentManager: NDManager,
    modelDimension: Int,
    headCount: Int,
    hiddenDimension: Int,
    maximumSequenceLength: Int,
    ropeTheta: Double = DEFAULT_ROPE_THETA,
    rmsNormEpsilon: Float = DEFAULT_RMS_NORM_EPSILON,
) : AbstractBlock(VERSION), AutoCloseable {
    val modelDimension: Int = requirePositive("modelDimension", modelDimension)
    val hiddenDimension: Int = requirePositive("hiddenDimension", hiddenDimension)

    val attentionNorm: RmsNorm = addChildBlock("attentionNorm", RmsNorm(this.modelDimension, rmsNormEpsilon))
    val attention: CausalSelfAttention = addChildBlock(
        "attention",
        CausalSelfAttention(
            parentManager,
            this.modelDimension,
            headCount,
            maximumSequenceLength,
            ropeTheta,
        ),
    )
    val feedForwardNorm: RmsNorm = addChildBlock("feedForwardNorm", RmsNorm(this.modelDimension, rmsNormEpsilon))
    val feedForward: SwiGluFeedForward = addChildBlock(
        "feedForward",
        SwiGluFeedForward(this.modelDimension, this.hiddenDimension),
    )

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList = NDList(
        forwardForInspection(
            parameterStore,
            inputs.singletonOrThrow(),
            training,
            includeResidualConnections = true,
        ).output,
    )

    fun forwardForInspection(
        parameterStore: ParameterStore,
        input: NDArray,
        training: Boolean,
        includeResidualConnections: Boolean = true,
    ): TransformerBlockResult {
        requireInputShape(input.shape)
        val normalizedAttentionInput = attentionNorm
            .forward(parameterStore, NDList(input), training)
            .singletonOrThrow()
        val attentionOutput = attention
            .forward(parameterStore, NDList(normalizedAttentionInput), training)
            .singletonOrThrow()
        val afterAttention = if (includeResidualConnections) input.add(attentionOutput) else attentionOutput
        val normalizedFeedForwardInput = feedForwardNorm
            .forward(parameterStore, NDList(afterAttention), training)
            .singletonOrThrow()
        val feedForwardResult = feedForward.forwardWithIntermediates(
            parameterStore,
            normalizedFeedForwardInput,
            training,
        )
        val feedForwardOutput = feedForwardResult.output
        val output = if (includeResidualConnections) afterAttention.add(feedForwardOutput) else feedForwardOutput
        return TransformerBlockResult(
            output,
            normalizedAttentionInput,
            attentionOutput,
            afterAttention,
            normalizedFeedForwardInput,
            feedForwardResult.hidden,
            feedForwardOutput,
        )
    }

    internal fun forwardIncremental(
        parameterStore: ParameterStore,
        input: NDArray,
        cache: AttentionKeyValueCache,
    ): NDArray {
        requireInputShape(input.shape)
        val normalizedAttentionInput = attentionNorm
            .forward(parameterStore, NDList(input), false)
            .singletonOrThrow()
        val attentionOutput = attention.forwardIncremental(parameterStore, normalizedAttentionInput, cache).output
        val afterAttention = input.add(attentionOutput)
        val normalizedFeedForwardInput = feedForwardNorm
            .forward(parameterStore, NDList(afterAttention), false)
            .singletonOrThrow()
        val feedForwardOutput = feedForward
            .forward(parameterStore, NDList(normalizedFeedForwardInput), false)
            .singletonOrThrow()
        return afterAttention.add(feedForwardOutput)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("TransformerBlock expects exactly one input shape")
        requireInputShape(inputShapes[0])
        return arrayOf(inputShapes[0])
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        if (inputShapes.size != 1) throw TensorShapeException("TransformerBlock expects exactly one input shape")
        requireInputShape(inputShapes[0])
        attentionNorm.initialize(manager, dataType, inputShapes[0])
        attention.initialize(manager, dataType, inputShapes[0])
        feedForwardNorm.initialize(manager, dataType, inputShapes[0])
        feedForward.initialize(manager, dataType, inputShapes[0])
    }

    fun parameterCount(): Long =
        attention.parameterCount() + feedForward.parameterCount() + 2L * modelDimension

    fun isRopeCacheOpen(): Boolean = attention.isRopeCacheOpen()

    override fun close() {
        try {
            clear()
        } finally {
            attention.close()
        }
    }

    private fun requireInputShape(shape: Shape) {
        if (shape.dimension() != EXPECTED_INPUT_RANK || shape.lastDimension != modelDimension.toLong()) {
            throw TensorShapeException("TransformerBlock expects [B, T, $modelDimension], got $shape")
        }
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw TensorShapeException("TransformerBlock requires positive batch and sequence dimensions, got $shape")
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 3
        const val DEFAULT_ROPE_THETA = 10_000.0
        const val DEFAULT_RMS_NORM_EPSILON = 1e-5f

        fun requirePositive(name: String, value: Int): Int {
            if (value <= 0) throw TensorShapeException("$name must be positive")
            return value
        }
    }
}

data class TransformerBlockResult(
    val output: NDArray,
    val normalizedAttentionInput: NDArray,
    val attentionOutput: NDArray,
    val afterAttentionResidual: NDArray,
    val normalizedFeedForwardInput: NDArray,
    val feedForwardHidden: NDArray,
    val feedForwardOutput: NDArray,
)
