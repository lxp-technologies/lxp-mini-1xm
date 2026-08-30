package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDList
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.initializer.NormalInitializer
import ai.djl.util.PairList

class TokenEmbedding(
    vocabularySize: Int,
    embeddingSize: Int,
    initializationStandardDeviation: Float = DEFAULT_INITIALIZATION_STANDARD_DEVIATION,
) : AbstractBlock(VERSION) {
    val vocabularySize: Int = requirePositive("vocabularySize", vocabularySize)
    val embeddingSize: Int = requirePositive("embeddingSize", embeddingSize)
    private val initializationStandardDeviation = requirePositiveStandardDeviation(initializationStandardDeviation)
    val weightParameter: Parameter = addParameter(
        Parameter.builder()
            .setName("weight")
            .setType(Parameter.Type.WEIGHT)
            .optShape(Shape(vocabularySize.toLong(), embeddingSize.toLong()))
            .optInitializer(NormalInitializer(initializationStandardDeviation))
            .build(),
    )

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val tokenIds = inputs.singletonOrThrow()
        if (tokenIds.shape.dimension() != EXPECTED_INPUT_RANK) {
            throw TensorShapeException("Token IDs must have shape [B, T], got ${tokenIds.shape}")
        }
        val weight = parameterStore.getValue(weightParameter, tokenIds.device, training)
        return NDList(weight.get(NDIndex("{}", tokenIds)))
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1 || inputShapes[0].dimension() != EXPECTED_INPUT_RANK) {
            throw TensorShapeException("TokenEmbedding expects one [B, T] input shape")
        }
        return arrayOf(inputShapes[0].add(embeddingSize.toLong()))
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 2
        const val DEFAULT_INITIALIZATION_STANDARD_DEVIATION = 0.02f

        fun requirePositive(name: String, value: Int): Int {
            if (value <= 0) throw TensorShapeException("$name must be positive")
            return value
        }

        fun requirePositiveStandardDeviation(value: Float): Float {
            if (!value.isFinite() || value <= 0f) {
                throw TensorShapeException("initializationStandardDeviation must be finite and positive")
            }
            return value
        }
    }
}
