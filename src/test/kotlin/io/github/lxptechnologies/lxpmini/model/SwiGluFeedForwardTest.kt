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
import kotlin.math.exp

class SwiGluFeedForwardTest {
    @Test
    fun `matches a manually calculated SwiGLU with identity projections`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val feedForward = SwiGluFeedForward(modelDimension = 2, hiddenDimension = 2)
            val input = manager.create(floatArrayOf(1f, 2f), Shape(1, 1, 2))
            feedForward.initialize(manager, DataType.FLOAT32, input.shape)
            val identity = floatArrayOf(1f, 0f, 0f, 1f)
            feedForward.gateWeight.array.set(identity)
            feedForward.valueWeight.array.set(identity)
            feedForward.downWeight.array.set(identity)

            val result = feedForward.forwardWithIntermediates(ParameterStore(manager, false), input, false)

            assertThat(result.gate.toFloatArray()).containsExactly(1f, 2f)
            assertThat(result.value.toFloatArray()).containsExactly(1f, 2f)
            assertThat(result.hidden.shape).isEqualTo(Shape(1, 1, 2))
            assertThat(result.output.getFloat(0, 0, 0))
                .isCloseTo((1.0 / (1.0 + exp(-1.0))).toFloat(), offset(1e-6f))
            assertThat(result.output.getFloat(0, 0, 1))
                .isCloseTo((4.0 / (1.0 + exp(-2.0))).toFloat(), offset(1e-6f))
            feedForward.clear()
        }
    }

    @Test
    fun `preserves outer shape and propagates finite gradients through three weights`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val feedForward = SwiGluFeedForward(modelDimension = 8, hiddenDimension = 16)
            val input = manager.randomNormal(0f, 1f, Shape(2, 4, 8), DataType.FLOAT32)
            input.setRequiresGradient(true)
            feedForward.initialize(manager, DataType.FLOAT32, input.shape)
            val parameterStore = ParameterStore(manager, false)

            val output = Engine.getInstance().newGradientCollector().use { collector ->
                feedForward.forward(parameterStore, NDList(input), true).singletonOrThrow().also {
                    collector.backward(it.square().mean())
                }
            }

            assertThat(output.shape).isEqualTo(input.shape)
            assertThat(input.gradient.toFloatArray().all(Float::isFinite)).isTrue()
            assertThat(feedForward.parameters.values()).hasSize(3).allSatisfy { parameter ->
                assertThat(parameter.array.gradient.toFloatArray().all(Float::isFinite)).isTrue()
            }
            assertThat(feedForward.parameterCount()).isEqualTo(384)
            feedForward.clear()
        }
    }

    @Test
    fun `rejects invalid dimensions and input shapes`() {
        assertThatThrownBy { SwiGluFeedForward(0, 8) }
            .isInstanceOf(TensorShapeException::class.java)
            .hasMessageContaining("modelDimension must be positive")
        assertThatThrownBy { SwiGluFeedForward(8, 0) }
            .isInstanceOf(TensorShapeException::class.java)
            .hasMessageContaining("hiddenDimension must be positive")

        val feedForward = SwiGluFeedForward(8, 16)
        assertThatThrownBy { feedForward.getOutputShapes(arrayOf(Shape(2, 8))) }
            .isInstanceOf(TensorShapeException::class.java)
            .hasMessageContaining("expects [B, T, 8]")
    }
}
