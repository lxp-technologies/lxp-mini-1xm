package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class CausalSelfAttentionTest {
    @Test
    fun `preserves model shape and exposes normalized triangular probabilities`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            CausalSelfAttention(manager, 8, 2, 16).use { attention ->
                val input = manager.randomNormal(0f, 1f, Shape(2, 4, 8), DataType.FLOAT32)
                attention.initialize(manager, DataType.FLOAT32, input.shape)

                val result = attention.forwardWithAttention(ParameterStore(manager, false), input, false)

                assertThat(result.output.shape).isEqualTo(Shape(2, 4, 8))
                assertThat(result.probabilities.shape).isEqualTo(Shape(2, 2, 4, 4))
                val probabilities = result.probabilities.toFloatArray()
                for (batch in 0 until 2) {
                    for (head in 0 until 2) {
                        for (query in 0 until 4) {
                            var rowSum = 0f
                            for (key in 0 until 4) {
                                val value = probabilities[((batch * 2 + head) * 4 + query) * 4 + key]
                                rowSum += value
                                if (key > query) assertThat(value).isZero()
                            }
                            assertThat(rowSum).isCloseTo(1f, offset(1e-6f))
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `changing a future token cannot change any past output`() {
        Engine.getInstance().setRandomSeed(42)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            CausalSelfAttention(manager, 8, 2, 16).use { attention ->
                val first = manager.randomNormal(0f, 1f, Shape(1, 4, 8), DataType.FLOAT32)
                val changedFuture = first.duplicate()
                changedFuture.set(ai.djl.ndarray.index.NDIndex("0, 3, :"), -999f)
                attention.initialize(manager, DataType.FLOAT32, first.shape)
                val parameterStore = ParameterStore(manager, false)

                val firstOutput = attention.forwardWithAttention(parameterStore, first, false).output
                val changedOutput = attention.forwardWithAttention(parameterStore, changedFuture, false).output

                assertThat(firstOutput.get("0, 0:3, :").toFloatArray())
                    .containsExactly(*changedOutput.get("0, 0:3, :").toFloatArray())
                assertThat(firstOutput.get("0, 3, :").toFloatArray())
                    .isNotEqualTo(changedOutput.get("0, 3, :").toFloatArray())
            }
        }
    }

    @Test
    fun `propagates finite gradients through input and four projections`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            CausalSelfAttention(manager, 8, 2, 16).use { attention ->
                val input = manager.randomNormal(0f, 1f, Shape(2, 4, 8), DataType.FLOAT32)
                input.setRequiresGradient(true)
                attention.initialize(manager, DataType.FLOAT32, input.shape)
                val parameterStore = ParameterStore(manager, false)

                Engine.getInstance().newGradientCollector().use { collector ->
                    val output = attention.forward(parameterStore, NDList(input), true).singletonOrThrow()
                    collector.backward(output.square().mean())
                }

                assertThat(input.hasGradient()).isTrue()
                assertThat(input.gradient.toFloatArray().all(Float::isFinite)).isTrue()
                assertThat(attention.parameters.values()).hasSize(4).allSatisfy { parameter ->
                    assertThat(parameter.array.shape).isEqualTo(Shape(8, 8))
                    assertThat(parameter.array.gradient.toFloatArray().all(Float::isFinite)).isTrue()
                }
                assertThat(attention.parameterCount()).isEqualTo(256)
            }
        }
    }

    @Test
    fun `rejects invalid dimensions and sequences beyond context`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            assertThatThrownBy { CausalSelfAttention(manager, 10, 2, 16) }
                .isInstanceOf(TensorShapeException::class.java)
                .hasMessageContaining("even for RoPE")

            CausalSelfAttention(manager, 8, 2, 3).use { attention ->
                assertThatThrownBy { attention.getOutputShapes(arrayOf(Shape(1, 4, 8))) }
                    .isInstanceOf(TensorShapeException::class.java)
                    .hasMessageContaining("exceeds maximum context 3")
            }
        }
    }

    @Test
    fun `closing attention releases its RoPE cache but not the parent manager`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val attention = CausalSelfAttention(manager, 8, 2, 16)

            assertThat(attention.isRopeCacheOpen()).isTrue()
            attention.close()

            assertThat(attention.isRopeCacheOpen()).isFalse()
            assertThat(manager.isOpen).isTrue()
        }
    }
}
