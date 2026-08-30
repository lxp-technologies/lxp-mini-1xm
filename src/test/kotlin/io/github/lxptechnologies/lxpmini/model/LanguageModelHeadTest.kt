package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class LanguageModelHeadTest {
    @Test
    fun `tied head uses the exact embedding parameter and its transposed array`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val embedding = TokenEmbedding(vocabularySize = 3, embeddingSize = 2)
            val head = LanguageModelHead(embedding.weightParameter, 2, 3, tieEmbeddings = true)
            embedding.initialize(manager, DataType.FLOAT32, Shape(1, 1))
            head.initialize(manager, DataType.FLOAT32, Shape(1, 1, 2))
            embedding.weightParameter.array.set(
                floatArrayOf(
                    1f, 0f,
                    0f, 1f,
                    1f, 1f,
                ),
            )
            val hidden = manager.create(floatArrayOf(1f, 2f), Shape(1, 1, 2))

            val logits = head.forward(ParameterStore(manager, false), NDList(hidden), false).singletonOrThrow()

            assertThat(head.weightParameter).isSameAs(embedding.weightParameter)
            assertThat(head.weightParameter.array).isSameAs(embedding.weightParameter.array)
            assertThat(head.directParameters).isEmpty()
            assertThat(head.additionalParameterCount()).isZero()
            assertThat(logits.shape).isEqualTo(Shape(1, 1, 3))
            assertThat(logits.toFloatArray()).containsExactly(1f, 2f, 3f)
            head.clear()
            embedding.clear()
        }
    }

    @Test
    fun `untied head owns an independent C by V projection`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val embedding = TokenEmbedding(vocabularySize = 3, embeddingSize = 2)
            val head = LanguageModelHead(embedding.weightParameter, 2, 3, tieEmbeddings = false)
            embedding.initialize(manager, DataType.FLOAT32, Shape(1, 1))
            head.initialize(manager, DataType.FLOAT32, Shape(1, 1, 2))

            assertThat(head.weightParameter).isNotSameAs(embedding.weightParameter)
            assertThat(head.weightParameter.array).isNotSameAs(embedding.weightParameter.array)
            assertThat(head.weightParameter.array.shape).isEqualTo(Shape(2, 3))
            assertThat(head.directParameters).hasSize(1)
            assertThat(head.additionalParameterCount()).isEqualTo(6)
            head.clear()
            embedding.clear()
        }
    }
}
