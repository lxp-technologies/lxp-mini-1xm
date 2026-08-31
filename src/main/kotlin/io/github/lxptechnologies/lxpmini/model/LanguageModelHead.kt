package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.util.PairList

class LanguageModelHead(
    embeddingWeight: Parameter,
    modelDimension: Int,
    vocabularySize: Int,
    tieEmbeddings: Boolean,
) : AbstractBlock(VERSION) {
    val modelDimension: Int = requirePositive("modelDimension", modelDimension)
    val vocabularySize: Int = requirePositive("vocabularySize", vocabularySize)
    val tieEmbeddings: Boolean = tieEmbeddings
    val weightParameter: Parameter = if (tieEmbeddings) {
        requireEmbeddingShape(embeddingWeight)
        embeddingWeight
    } else {
        addParameter(
            Parameter.builder()
                .setName("weight")
                .setType(Parameter.Type.WEIGHT)
                .optShape(Shape(modelDimension.toLong(), vocabularySize.toLong()))
                .optInitializer(NormalInitializer(INITIALIZATION_STANDARD_DEVIATION))
                .build(),
        )
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val input = inputs.singletonOrThrow()
        requireInputShape(input.shape)
        val weight = parameterStore.getValue(weightParameter, input.device, training)
        val projectionWeight = if (tieEmbeddings) {
            weight.get(input.manager, NDIndex(":, :")).transpose()
        } else {
            weight
        }
        return NDList(input.matMul(projectionWeight))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("LanguageModelHead expects exactly one input shape")
        requireInputShape(inputShapes[0])
        val input = inputShapes[0]
        return arrayOf(Shape(input[0], input[1], vocabularySize.toLong()))
    }

    fun additionalParameterCount(): Long = if (tieEmbeddings) 0 else modelDimension.toLong() * vocabularySize

    private fun requireEmbeddingShape(parameter: Parameter) {
        val expected = Shape(vocabularySize.toLong(), modelDimension.toLong())
        if (parameter.shape != expected) {
            throw TensorShapeException("Tied embedding weight must have shape $expected, got ${parameter.shape}")
        }
    }

    private fun requireInputShape(shape: Shape) {
        if (shape.dimension() != EXPECTED_INPUT_RANK || shape.lastDimension != modelDimension.toLong()) {
            throw TensorShapeException("LanguageModelHead expects [B, T, $modelDimension], got $shape")
        }
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw TensorShapeException("LanguageModelHead requires positive batch and sequence dimensions, got $shape")
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 3
        const val INITIALIZATION_STANDARD_DEVIATION = 0.02f

        fun requirePositive(name: String, value: Int): Int {
            if (value <= 0) throw TensorShapeException("$name must be positive")
            return value
        }
    }
}
