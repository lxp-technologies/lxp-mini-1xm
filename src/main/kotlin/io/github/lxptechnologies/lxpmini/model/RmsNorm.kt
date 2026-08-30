package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.Initializer
import ai.djl.util.PairList

class RmsNorm(
    modelDimension: Int,
    epsilon: Float = DEFAULT_EPSILON,
) : AbstractBlock(VERSION) {
    val modelDimension: Int = requirePositiveDimension(modelDimension)
    val epsilon: Float = requirePositiveEpsilon(epsilon)
    val scaleParameter: Parameter = addParameter(
        Parameter.builder()
            .setName("scale")
            .setType(Parameter.Type.GAMMA)
            .optShape(Shape(modelDimension.toLong()))
            .optInitializer(Initializer.ONES)
            .build(),
    )

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val input = inputs.singletonOrThrow()
        requireLastDimension(input.shape)
        val scale = parameterStore.getValue(scaleParameter, input.device, training)
        val meanSquare = input.square().mean(intArrayOf(input.shape.dimension() - 1), true)
        val inverseRootMeanSquare = meanSquare.add(epsilon).sqrt()
        return NDList(input.div(inverseRootMeanSquare).mul(scale))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("RmsNorm expects exactly one input shape")
        requireLastDimension(inputShapes[0])
        return arrayOf(inputShapes[0])
    }

    private fun requireLastDimension(shape: Shape) {
        if (shape.dimension() == 0 || shape.lastDimension != modelDimension.toLong()) {
            throw TensorShapeException("RmsNorm expects a final dimension of $modelDimension, got $shape")
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val DEFAULT_EPSILON = 1e-5f

        fun requirePositiveDimension(value: Int): Int {
            if (value <= 0) throw TensorShapeException("modelDimension must be positive")
            return value
        }

        fun requirePositiveEpsilon(value: Float): Float {
            if (!value.isFinite() || value <= 0f) throw TensorShapeException("epsilon must be finite and positive")
            return value
        }
    }
}
