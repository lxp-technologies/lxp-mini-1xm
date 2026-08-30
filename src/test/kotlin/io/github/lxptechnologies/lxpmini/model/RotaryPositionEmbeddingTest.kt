package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.cos
import kotlin.math.sin

class RotaryPositionEmbeddingTest {
    @Test
    fun `keeps position zero unchanged and rotates adjacent pairs at position one`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            RotaryPositionEmbedding(manager, headDimension = 4, maximumSequenceLength = 8).use { rope ->
                val input = manager.create(
                    floatArrayOf(
                        1f, 2f, 3f, 4f,
                        1f, 0f, 1f, 0f,
                    ),
                    Shape(1, 1, 2, 4),
                )

                val output = rope.apply(input).toFloatArray()

                assertThat(output[0]).isCloseTo(1f, offset(1e-6f))
                assertThat(output[1]).isCloseTo(2f, offset(1e-6f))
                assertThat(output[2]).isCloseTo(3f, offset(1e-6f))
                assertThat(output[3]).isCloseTo(4f, offset(1e-6f))
                assertThat(output[4]).isCloseTo(cos(1.0).toFloat(), offset(1e-6f))
                assertThat(output[5]).isCloseTo(sin(1.0).toFloat(), offset(1e-6f))
                assertThat(output[6]).isCloseTo(cos(0.01).toFloat(), offset(1e-6f))
                assertThat(output[7]).isCloseTo(sin(0.01).toFloat(), offset(1e-6f))
            }
        }
    }

    @Test
    fun `preserves shape and propagates gradients through the rotation`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            RotaryPositionEmbedding(manager, headDimension = 4, maximumSequenceLength = 8).use { rope ->
                val input = manager.ones(Shape(2, 3, 4, 4))
                input.setRequiresGradient(true)

                val output = Engine.getInstance().newGradientCollector().use { collector ->
                    rope.apply(input).also { rotated -> collector.backward(rotated.sum()) }
                }

                assertThat(output.shape).isEqualTo(input.shape)
                assertThat(input.hasGradient()).isTrue()
                assertThat(input.gradient.toFloatArray().all(Float::isFinite)).isTrue()
                assertThat(rope.cacheShape()).isEqualTo(Shape(8, 2))
            }
        }
    }

    @Test
    fun `supports offsets and rejects positions outside the cache`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            RotaryPositionEmbedding(manager, headDimension = 2, maximumSequenceLength = 4).use { rope ->
                val input = manager.create(floatArrayOf(1f, 0f), Shape(1, 1, 1, 2))
                val atPositionTwo = rope.apply(input, startPosition = 2).toFloatArray()

                assertThat(atPositionTwo[0]).isCloseTo(cos(2.0).toFloat(), offset(1e-6f))
                assertThat(atPositionTwo[1]).isCloseTo(sin(2.0).toFloat(), offset(1e-6f))
                assertThatThrownBy { rope.apply(input, startPosition = 4) }
                    .isInstanceOf(TensorShapeException::class.java)
                    .hasMessageContaining("exceed context 4")
            }
        }
    }

    @Test
    fun `closes only its cache submanager`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val rope = RotaryPositionEmbedding(manager, headDimension = 4, maximumSequenceLength = 8)

            assertThat(rope.isOpen()).isTrue()
            rope.close()

            assertThat(rope.isOpen()).isFalse()
            assertThat(manager.isOpen).isTrue()
        }
    }

    @Test
    fun `validates dimensions before allocating a cache submanager`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val childCountBefore = manager.managedArrays.size

            assertThatThrownBy {
                RotaryPositionEmbedding(manager, headDimension = 3, maximumSequenceLength = 8)
            }.isInstanceOf(TensorShapeException::class.java)
                .hasMessageContaining("positive and even")

            assertThat(manager.isOpen).isTrue()
            assertThat(manager.managedArrays).hasSize(childCountBefore)
        }
    }
}
