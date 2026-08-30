package io.github.lxptechnologies.lxpmini.checkpoint

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.training.TrainingProgress
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class CheckpointStoreTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    private val store = CheckpointStore()
    private val configChecksum = "a".repeat(64)

    @Test
    fun `round trip recreates identical logits and trainable parameters`() {
        val shape = Shape(1, 4)
        val tokenValues = longArrayOf(1, 2, 3, 4)
        val expectedLogits = NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, shape)
                val logits = forward(model, manager, tokenValues, shape)
                val saved = store.save(
                    temporaryDirectory,
                    model,
                    TrainingProgress(optimizerUpdates = 3, tokensSeen = 12),
                    totalUpdates = 10,
                    configSha256 = configChecksum,
                )
                assertThat(saved.manifest.exactTrainingResume).isFalse()
                assertThat(saved.manifest.optimizerMomentsRestored).isFalse()
                logits
            }
        }

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                val loaded = store.loadLatest(temporaryDirectory, model, manager, configChecksum)
                val actualLogits = forward(model, manager, tokenValues, shape)

                assertThat(actualLogits).containsExactly(*expectedLogits)
                assertThat(loaded.progress).isEqualTo(TrainingProgress(3, 12))
                Engine.getInstance().newGradientCollector().use { collector ->
                    val tokenIds = manager.create(tokenValues, shape)
                    val output = model.forward(ParameterStore(manager, false), NDList(tokenIds), true).singletonOrThrow()
                    collector.backward(output.sum())
                }
                assertThat(model.parameters.values().all { parameter -> parameter.array.hasGradient() }).isTrue()
            }
        }
    }

    @Test
    fun `rejects corrupted model parameters before loading`() {
        saveExample()
        val modelPath = temporaryDirectory.resolve("checkpoints/step-00000003/model.params")
        val bytes = Files.readAllBytes(modelPath)
        bytes[bytes.lastIndex] = (bytes.last().toInt() xor 0x01).toByte()
        Files.write(modelPath, bytes)

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                assertThatThrownBy { store.loadLatest(temporaryDirectory, model, manager, configChecksum) }
                    .isInstanceOf(CheckpointException::class.java)
                    .hasMessageContaining("checksum mismatch")
            }
        }
    }

    @Test
    fun `rejects a checkpoint associated with another configuration`() {
        saveExample()

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                assertThatThrownBy { store.loadLatest(temporaryDirectory, model, manager, "b".repeat(64)) }
                    .isInstanceOf(CheckpointException::class.java)
                    .hasMessageContaining("configuration checksum")
            }
        }
    }

    private fun saveExample() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))
                store.save(temporaryDirectory, model, TrainingProgress(3, 12), 10, configChecksum)
            }
        }
    }

    private fun forward(
        model: DecoderLanguageModel,
        manager: NDManager,
        tokenValues: LongArray,
        shape: Shape,
    ): FloatArray {
        val tokenIds = manager.create(tokenValues, shape)
        return model.forward(ParameterStore(manager, false), NDList(tokenIds), false).singletonOrThrow().toFloatArray()
    }

    private fun tinyModelConfig() = ModelConfig(
        vocabSize = 16,
        contextLength = 4,
        dModel = 8,
        numLayers = 1,
        numHeads = 2,
        ffnDim = 16,
        ropeTheta = 10_000.0,
        dropout = 0.0,
        tieEmbeddings = true,
    )
}
