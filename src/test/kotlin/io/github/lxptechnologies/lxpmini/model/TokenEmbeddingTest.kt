package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class TokenEmbeddingTest {
    @Test
    fun `maps each token ID to its learned row and preserves B T dimensions`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val embedding = TokenEmbedding(vocabularySize = 4, embeddingSize = 3)
            embedding.initialize(manager, DataType.FLOAT32, Shape(2, 2))
            embedding.weightParameter.array.set(
                floatArrayOf(
                    0f, 1f, 2f,
                    10f, 11f, 12f,
                    20f, 21f, 22f,
                    30f, 31f, 32f,
                ),
            )
            val tokenIds = manager.create(longArrayOf(2, 0, 3, 1), Shape(2, 2))

            val output = embedding.forward(ParameterStore(manager, false), NDList(tokenIds), false).singletonOrThrow()

            assertThat(output.shape).isEqualTo(Shape(2, 2, 3))
            assertThat(output.toFloatArray()).containsExactly(
                20f, 21f, 22f,
                0f, 1f, 2f,
                30f, 31f, 32f,
                10f, 11f, 12f,
            )
            embedding.clear()
        }
    }

    @Test
    fun `propagates gradients only through rows selected by token IDs`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val embedding = TokenEmbedding(vocabularySize = 4, embeddingSize = 2)
            embedding.initialize(manager, DataType.FLOAT32, Shape(1, 3))
            val tokenIds = manager.create(longArrayOf(1, 1, 3), Shape(1, 3))

            Engine.getInstance().newGradientCollector().use { collector ->
                val output = embedding.forward(ParameterStore(manager, false), NDList(tokenIds), true).singletonOrThrow()
                collector.backward(output.sum())
            }

            val gradient = embedding.weightParameter.array.gradient.toFloatArray()
            assertThat(gradient).containsExactly(
                0f, 0f,
                2f, 2f,
                0f, 0f,
                1f, 1f,
            )
            assertThat(gradient.sum()).isCloseTo(6f, offset(1e-6f))
            embedding.clear()
        }
    }
}
