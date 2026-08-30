package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class TransformerBlockTest {
    @Test
    fun `preserves shape stays finite and has the expected parameter count`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            TransformerBlock(manager, 8, 2, 16, 16).use { block ->
                val input = manager.randomNormal(0f, 1f, Shape(2, 4, 8), DataType.FLOAT32)
                block.initialize(manager, DataType.FLOAT32, input.shape)

                val result = block.forwardForInspection(ParameterStore(manager, false), input, false)

                assertThat(result.output.shape).isEqualTo(input.shape)
                assertThat(result.normalizedAttentionInput.shape).isEqualTo(input.shape)
                assertThat(result.attentionOutput.shape).isEqualTo(input.shape)
                assertThat(result.afterAttentionResidual.shape).isEqualTo(input.shape)
                assertThat(result.normalizedFeedForwardInput.shape).isEqualTo(input.shape)
                assertThat(result.feedForwardHidden.shape).isEqualTo(Shape(2, 4, 16))
                assertThat(result.feedForwardOutput.shape).isEqualTo(input.shape)
                assertThat(result.output.toFloatArray().all(Float::isFinite)).isTrue()
                assertThat(block.parameters.values()).hasSize(9)
                assertThat(block.parameters.values().sumOf { it.array.size() }).isEqualTo(656)
                assertThat(block.parameterCount()).isEqualTo(656)
            }
        }
    }

    @Test
    fun `residual paths make zeroed sublayers an exact identity`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            TransformerBlock(manager, 8, 2, 16, 16).use { block ->
                val input = manager.randomNormal(0f, 1f, Shape(1, 4, 8), DataType.FLOAT32)
                block.initialize(manager, DataType.FLOAT32, input.shape)
                block.attention.outputWeight.array.set(FloatArray(8 * 8))
                block.feedForward.downWeight.array.set(FloatArray(16 * 8))
                val parameterStore = ParameterStore(manager, false)

                val withResiduals = block.forwardForInspection(parameterStore, input, false, true).output
                val withoutResiduals = block.forwardForInspection(parameterStore, input, false, false).output

                assertThat(withResiduals.toFloatArray()).containsExactly(*input.toFloatArray())
                assertThat(withoutResiduals.toFloatArray()).containsOnly(0f)
            }
        }
    }

    @Test
    fun `changing a future token cannot change past block outputs`() {
        Engine.getInstance().setRandomSeed(42)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            TransformerBlock(manager, 8, 2, 16, 16).use { block ->
                val input = manager.randomNormal(0f, 1f, Shape(1, 4, 8), DataType.FLOAT32)
                val changedFuture = input.duplicate()
                changedFuture.set(NDIndex("0, 3, :"), 999f)
                block.initialize(manager, DataType.FLOAT32, input.shape)
                val parameterStore = ParameterStore(manager, false)

                val originalOutput = block.forward(parameterStore, NDList(input), false).singletonOrThrow()
                val changedOutput = block.forward(parameterStore, NDList(changedFuture), false).singletonOrThrow()

                assertThat(originalOutput.get("0, 0:3, :").toFloatArray())
                    .containsExactly(*changedOutput.get("0, 0:3, :").toFloatArray())
                assertThat(originalOutput.get("0, 3, :").toFloatArray())
                    .isNotEqualTo(changedOutput.get("0, 3, :").toFloatArray())
            }
        }
    }

    @Test
    fun `propagates different finite gradients with and without residual paths`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            TransformerBlock(manager, 8, 2, 16, 16).use { block ->
                val values = FloatArray(32) { index -> (index + 1) / 32f }
                val withResidualsInput = manager.create(values, Shape(1, 4, 8))
                val withoutResidualsInput = manager.create(values, Shape(1, 4, 8))
                withResidualsInput.setRequiresGradient(true)
                withoutResidualsInput.setRequiresGradient(true)
                block.initialize(manager, DataType.FLOAT32, withResidualsInput.shape)
                val parameterStore = ParameterStore(manager, false)

                Engine.getInstance().newGradientCollector().use { collector ->
                    val output = block.forwardForInspection(parameterStore, withResidualsInput, true, true).output
                    collector.backward(output.square().mean())
                }
                Engine.getInstance().newGradientCollector().use { collector ->
                    val output = block.forwardForInspection(parameterStore, withoutResidualsInput, true, false).output
                    collector.backward(output.square().mean())
                }

                val withGradient = withResidualsInput.gradient.toFloatArray()
                val withoutGradient = withoutResidualsInput.gradient.toFloatArray()
                assertThat(withGradient.all(Float::isFinite)).isTrue()
                assertThat(withoutGradient.all(Float::isFinite)).isTrue()
                assertThat(withGradient).isNotEqualTo(withoutGradient)
            }
        }
    }

    @Test
    fun `closing block releases the attention cache but leaves its parent manager open`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val block = TransformerBlock(manager, 8, 2, 16, 16)

            assertThat(block.isRopeCacheOpen()).isTrue()
            block.close()

            assertThat(block.isRopeCacheOpen()).isFalse()
            assertThat(manager.isOpen).isTrue()
        }
    }
}
