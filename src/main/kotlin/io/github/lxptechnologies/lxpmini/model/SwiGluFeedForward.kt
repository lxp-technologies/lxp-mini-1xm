package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.XavierInitializer
import ai.djl.util.PairList

class SwiGluFeedForward(
    modelDimension: Int,
    hiddenDimension: Int,
) : AbstractBlock(VERSION) {
    val modelDimension: Int = requirePositive("modelDimension", modelDimension)
    val hiddenDimension: Int = requirePositive("hiddenDimension", hiddenDimension)

    val gateWeight: Parameter = projectionParameter(
        "gateWeight",
        Shape(this.modelDimension.toLong(), this.hiddenDimension.toLong()),
    )
    val valueWeight: Parameter = projectionParameter(
        "valueWeight",
        Shape(this.modelDimension.toLong(), this.hiddenDimension.toLong()),
    )
    val downWeight: Parameter = projectionParameter(
        "downWeight",
        Shape(this.hiddenDimension.toLong(), this.modelDimension.toLong()),
    )

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList = NDList(forwardWithIntermediates(parameterStore, inputs.singletonOrThrow(), training).output)

    fun forwardWithIntermediates(
        parameterStore: ParameterStore,
        input: NDArray,
        training: Boolean,
    ): SwiGluResult {
        requireInputShape(input.shape)
        val gate = project(input, gateWeight, parameterStore, training)
        val value = project(input, valueWeight, parameterStore, training)
        val activatedGate = silu(gate)
        val hidden = activatedGate.mul(value)
        val output = project(hidden, downWeight, parameterStore, training)
        return SwiGluResult(output, gate, value, activatedGate, hidden)
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("SwiGluFeedForward expects exactly one input shape")
        requireInputShape(inputShapes[0])
        return arrayOf(inputShapes[0])
    }

    fun parameterCount(): Long = 3L * modelDimension * hiddenDimension

    private fun projectionParameter(name: String, shape: Shape): Parameter = addParameter(
        Parameter.builder()
            .setName(name)
            .setType(Parameter.Type.WEIGHT)
            .optShape(shape)
            .optInitializer(XavierInitializer())
            .build(),
    )

    private fun project(
        input: NDArray,
        parameter: Parameter,
        parameterStore: ParameterStore,
        training: Boolean,
    ): NDArray = input.matMul(parameterStore.getValue(parameter, input.device, training))

    private fun silu(input: NDArray): NDArray {
        val sigmoid = input.neg().exp().add(1f).pow(-1f)
        return input.mul(sigmoid)
    }

    private fun requireInputShape(shape: Shape) {
        if (shape.dimension() != EXPECTED_INPUT_RANK || shape.lastDimension != modelDimension.toLong()) {
            throw TensorShapeException("SwiGluFeedForward expects [B, T, $modelDimension], got $shape")
        }
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw TensorShapeException("SwiGluFeedForward requires positive batch and sequence dimensions, got $shape")
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 3

        fun requirePositive(name: String, value: Int): Int {
            if (value <= 0) throw TensorShapeException("$name must be positive")
            return value
        }
    }
}

data class SwiGluResult(
    val output: NDArray,
    val gate: NDArray,
    val value: NDArray,
    val activatedGate: NDArray,
    val hidden: NDArray,
)
