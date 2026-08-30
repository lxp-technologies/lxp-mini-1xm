package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.XavierInitializer
import ai.djl.util.PairList
import kotlin.math.sqrt

class CausalSelfAttention(
    parentManager: ai.djl.ndarray.NDManager,
    modelDimension: Int,
    headCount: Int,
    maximumSequenceLength: Int,
    ropeTheta: Double = DEFAULT_ROPE_THETA,
) : AbstractBlock(VERSION), AutoCloseable {
    val modelDimension: Int = requirePositive("modelDimension", modelDimension)
    val headCount: Int = requirePositive("headCount", headCount)
    val maximumSequenceLength: Int = requirePositive("maximumSequenceLength", maximumSequenceLength)
    val headDimension: Int = requireHeadDimension(this.modelDimension, this.headCount)
    private val rope = RotaryPositionEmbedding(
        parentManager,
        headDimension,
        this.maximumSequenceLength,
        ropeTheta,
    )

    val queryWeight: Parameter = projectionParameter("queryWeight")
    val keyWeight: Parameter = projectionParameter("keyWeight")
    val valueWeight: Parameter = projectionParameter("valueWeight")
    val outputWeight: Parameter = projectionParameter("outputWeight")

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList = NDList(forwardWithAttention(parameterStore, inputs.singletonOrThrow(), training).output)

    fun forwardWithAttention(
        parameterStore: ParameterStore,
        input: NDArray,
        training: Boolean,
    ): CausalAttentionResult {
        requireInputShape(input.shape)
        val query = rope.apply(splitHeads(project(input, queryWeight, parameterStore, training)))
        val key = rope.apply(splitHeads(project(input, keyWeight, parameterStore, training)))
        val value = splitHeads(project(input, valueWeight, parameterStore, training))

        val scores = query.matMul(key.transpose(0, 1, 3, 2)).div(sqrt(headDimension.toDouble()))
        val probabilities = scores.add(causalMask(input, input.shape[1])).softmax(-1)
        val context = probabilities.matMul(value)
        val merged = mergeHeads(context)
        val output = project(merged, outputWeight, parameterStore, training)
        return CausalAttentionResult(output, probabilities)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("CausalSelfAttention expects exactly one input shape")
        requireInputShape(inputShapes[0])
        return arrayOf(inputShapes[0])
    }

    fun parameterCount(): Long = 4L * modelDimension * modelDimension

    fun isRopeCacheOpen(): Boolean = rope.isOpen()

    override fun close() {
        try {
            clear()
        } finally {
            rope.close()
        }
    }

    private fun projectionParameter(name: String): Parameter = addParameter(
        Parameter.builder()
            .setName(name)
            .setType(Parameter.Type.WEIGHT)
            .optShape(Shape(modelDimension.toLong(), modelDimension.toLong()))
            .optInitializer(XavierInitializer())
            .build(),
    )

    private fun project(
        input: NDArray,
        parameter: Parameter,
        parameterStore: ParameterStore,
        training: Boolean,
    ): NDArray = input.matMul(parameterStore.getValue(parameter, input.device, training))

    private fun splitHeads(input: NDArray): NDArray {
        val batchSize = input.shape[0]
        val sequenceLength = input.shape[1]
        return input
            .reshape(batchSize, sequenceLength, headCount.toLong(), headDimension.toLong())
            .transpose(0, 2, 1, 3)
    }

    private fun mergeHeads(input: NDArray): NDArray {
        val batchSize = input.shape[0]
        val sequenceLength = input.shape[2]
        return input.transpose(0, 2, 1, 3).reshape(batchSize, sequenceLength, modelDimension.toLong())
    }

    private fun causalMask(input: NDArray, sequenceLength: Long): NDArray {
        val elementCount = sequenceLength * sequenceLength
        if (elementCount > Int.MAX_VALUE) {
            throw TensorShapeException("Attention mask exceeds the JVM array limit")
        }
        val size = sequenceLength.toInt()
        val values = FloatArray(elementCount.toInt())
        for (queryPosition in 0 until size) {
            for (keyPosition in queryPosition + 1 until size) {
                values[queryPosition * size + keyPosition] = Float.NEGATIVE_INFINITY
            }
        }
        return input.manager.create(values, Shape(sequenceLength, sequenceLength))
    }

    private fun requireInputShape(shape: Shape) {
        if (shape.dimension() != EXPECTED_INPUT_RANK || shape.lastDimension != modelDimension.toLong()) {
            throw TensorShapeException("CausalSelfAttention expects [B, T, $modelDimension], got $shape")
        }
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw TensorShapeException("CausalSelfAttention requires positive batch and sequence dimensions, got $shape")
        }
        if (shape[1] > maximumSequenceLength) {
            throw TensorShapeException(
                "Sequence length ${shape[1]} exceeds maximum context $maximumSequenceLength",
            )
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 3
        const val DEFAULT_ROPE_THETA = 10_000.0

        fun requirePositive(name: String, value: Int): Int {
            if (value <= 0) throw TensorShapeException("$name must be positive")
            return value
        }

        fun requireHeadDimension(modelDimension: Int, headCount: Int): Int {
            if (modelDimension % headCount != 0) {
                throw TensorShapeException("modelDimension must be divisible by headCount")
            }
            val headDimension = modelDimension / headCount
            if (headDimension % 2 != 0) {
                throw TensorShapeException("headDimension must be even for RoPE")
            }
            return headDimension
        }
    }
}

data class CausalAttentionResult(
    val output: NDArray,
    val probabilities: NDArray,
)
