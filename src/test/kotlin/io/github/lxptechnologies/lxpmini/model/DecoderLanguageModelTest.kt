package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import java.nio.file.Path

class DecoderLanguageModelTest {
    private val parameterCounter = ParameterCounter()

    @Test
    fun `produces finite B by T by V logits with the theoretical tied count`() {
        val config = tinyConfig(tieEmbeddings = true)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config).use { model ->
                val tokenIds = manager.create(longArrayOf(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 4))
                model.initialize(manager, DataType.FLOAT32, tokenIds.shape)

                val result = model.forwardWithIntermediates(ParameterStore(manager, false), tokenIds, false)

                assertThat(result.embeddings.shape).isEqualTo(Shape(2, 4, 8))
                assertThat(result.blockOutputs).hasSize(2).allSatisfy { output ->
                    assertThat(output.shape).isEqualTo(Shape(2, 4, 8))
                }
                assertThat(result.normalizedOutput.shape).isEqualTo(Shape(2, 4, 8))
                assertThat(result.logits.shape).isEqualTo(Shape(2, 4, 16))
                assertThat(result.logits.toFloatArray().all(Float::isFinite)).isTrue()
                assertThat(model.sharesEmbeddingParameter()).isTrue()
                assertThat(model.sharesEmbeddingArray()).isTrue()
                assertThat(model.parameterTensorCount()).isEqualTo(20)
                assertThat(model.actualParameterCount()).isEqualTo(parameterCounter.count(config).total)
                assertThat(model.actualParameterCount()).isEqualTo(1_448)
            }
        }
    }

    @Test
    fun `untied head adds V by C weights and does not share objects`() {
        val tiedConfig = tinyConfig(tieEmbeddings = true)
        val untiedConfig = tiedConfig.copy(tieEmbeddings = false)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, untiedConfig).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))

                assertThat(model.sharesEmbeddingParameter()).isFalse()
                assertThat(model.sharesEmbeddingArray()).isFalse()
                assertThat(model.parameterTensorCount()).isEqualTo(21)
                assertThat(model.actualParameterCount()).isEqualTo(parameterCounter.count(untiedConfig).total)
                assertThat(model.actualParameterCount() - parameterCounter.count(tiedConfig).total).isEqualTo(128)
            }
        }
    }

    @Test
    fun `changing a future token cannot change any past logit`() {
        Engine.getInstance().setRandomSeed(42)
        val config = tinyConfig()
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config).use { model ->
                val tokens = manager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                val changedFuture = tokens.duplicate()
                changedFuture.set(NDIndex("0, 3"), 15)
                model.initialize(manager, DataType.FLOAT32, tokens.shape)
                val parameterStore = ParameterStore(manager, false)

                val original = model.forward(parameterStore, NDList(tokens), false).singletonOrThrow()
                val changed = model.forward(parameterStore, NDList(changedFuture), false).singletonOrThrow()

                assertThat(original.get("0, 0:3, :").toFloatArray())
                    .containsExactly(*changed.get("0, 0:3, :").toFloatArray())
                assertThat(original.get("0, 3, :").toFloatArray())
                    .isNotEqualTo(changed.get("0, 3, :").toFloatArray())
            }
        }
    }

    @Test
    fun `backpropagates finite gradients from logits through every registered parameter`() {
        val config = tinyConfig()
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config).use { model ->
                val tokens = manager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                model.initialize(manager, DataType.FLOAT32, tokens.shape)
                val parameterStore = ParameterStore(manager, false)

                Engine.getInstance().newGradientCollector().use { collector ->
                    val logits = model.forward(parameterStore, NDList(tokens), true).singletonOrThrow()
                    collector.backward(logits.square().mean())
                }

                assertThat(model.parameters.values()).allSatisfy { parameter ->
                    assertThat(parameter.array.hasGradient()).isTrue()
                    assertThat(parameter.array.gradient.toFloatArray().all(Float::isFinite)).isTrue()
                }
            }
        }
    }

    @Test
    fun `mini 17m initializes exactly the documented number of parameters`() {
        val config = ConfigLoader().load(Path.of("configs/mini-17m.yaml")).model
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))

                assertThat(model.actualParameterCount()).isEqualTo(17_308_032)
                assertThat(model.actualParameterCount()).isEqualTo(parameterCounter.count(config).total)
                assertThat(model.parameterTensorCount()).isEqualTo(74)
                assertThat(model.sharesEmbeddingArray()).isTrue()
                assertThat(model.openRopeCacheCount()).isEqualTo(8)
            }
        }
    }

    @Test
    fun `closing model releases every RoPE cache but leaves parent manager open`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val model = DecoderLanguageModel(manager, tinyConfig())

            assertThat(model.openRopeCacheCount()).isEqualTo(2)
            model.close()

            assertThat(model.openRopeCacheCount()).isZero()
            assertThat(manager.isOpen).isTrue()
        }
    }

    @Test
    fun `rejects unsupported dropout and sequences beyond context`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            assertThatThrownBy { DecoderLanguageModel(manager, tinyConfig().copy(dropout = 0.1)) }
                .isInstanceOf(TensorShapeException::class.java)
                .hasMessageContaining("dropout is not implemented")

            DecoderLanguageModel(manager, tinyConfig()).use { model ->
                assertThatThrownBy { model.getOutputShapes(arrayOf(Shape(1, 9))) }
                    .isInstanceOf(TensorShapeException::class.java)
                    .hasMessageContaining("exceeds context 8")
            }
        }
    }

    private fun tinyConfig(tieEmbeddings: Boolean = true) = ModelConfig(
        vocabSize = 16,
        contextLength = 8,
        dModel = 8,
        numLayers = 2,
        numHeads = 2,
        ffnDim = 16,
        ropeTheta = 10_000.0,
        dropout = 0.0,
        tieEmbeddings = tieEmbeddings,
    )
}
