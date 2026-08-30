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
import kotlin.math.sqrt

class RmsNormTest {
    @Test
    fun `matches the manual RMS formula on the final dimension`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val norm = RmsNorm(modelDimension = 2, epsilon = 1e-5f)
            norm.initialize(manager, DataType.FLOAT32, Shape(1, 1, 2))
            norm.scaleParameter.array.set(floatArrayOf(2f, 0.5f))
            val input = manager.create(floatArrayOf(3f, 4f), Shape(1, 1, 2))

            val output = norm.forward(ParameterStore(manager, false), NDList(input), false).singletonOrThrow()
            val denominator = sqrt(12.5f + 1e-5f)

            assertThat(output.shape).isEqualTo(input.shape)
            assertThat(output.toFloatArray()[0]).isCloseTo(3f / denominator * 2f, offset(1e-6f))
            assertThat(output.toFloatArray()[1]).isCloseTo(4f / denominator * 0.5f, offset(1e-6f))
            norm.clear()
        }
    }

    @Test
    fun `propagates gradients to the input and learned scale`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val norm = RmsNorm(modelDimension = 2)
            norm.initialize(manager, DataType.FLOAT32, Shape(1, 1, 2))
            val input = manager.create(floatArrayOf(1f, 2f), Shape(1, 1, 2))
            input.setRequiresGradient(true)

            Engine.getInstance().newGradientCollector().use { collector ->
                val output = norm.forward(ParameterStore(manager, false), NDList(input), true).singletonOrThrow()
                collector.backward(output.sum())
            }

            assertThat(input.hasGradient()).isTrue()
            assertThat(input.gradient.toFloatArray().all(Float::isFinite)).isTrue()
            assertThat(norm.scaleParameter.array.hasGradient()).isTrue()
            assertThat(norm.scaleParameter.array.gradient.toFloatArray().all(Float::isFinite)).isTrue()
            norm.clear()
        }
    }

    @Test
    fun `rejects a tensor whose final dimension differs`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val norm = RmsNorm(modelDimension = 4)
            norm.initialize(manager, DataType.FLOAT32, Shape(1, 2, 4))
            val invalid = manager.zeros(Shape(1, 2, 3))

            assertThatThrownBy {
                norm.forward(ParameterStore(manager, false), NDList(invalid), false)
            }.isInstanceOf(TensorShapeException::class.java)
                .hasMessageContaining("final dimension of 4")
            norm.clear()
        }
    }
}
