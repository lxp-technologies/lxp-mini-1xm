package io.github.lxptechnologies.lxpmini.training

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import io.github.lxptechnologies.lxpmini.config.TrainingConfig
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class LanguageModelTrainerTest {
    @Test
    fun `accumulates gradients before one clipped AdamW update`() {
        Engine.getInstance().setRandomSeed(42)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))
                val trainer = LanguageModelTrainer(
                    model,
                    manager,
                    tinyTrainingConfig(accumulationSteps = 2, gradientClipNorm = 1e-5, warmupSteps = 1),
                    totalUpdates = 4,
                )
                val originalWeight = model.embedding.weightParameter.array.toFloatArray()
                val input = manager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                val targets = manager.create(longArrayOf(2, 3, 4, 5), Shape(1, 4))

                val first = trainer.trainMicroBatch(input, targets)
                assertThat(first.optimizerUpdate).isNull()
                assertThat(model.embedding.weightParameter.array.toFloatArray()).containsExactly(*originalWeight)

                val second = trainer.trainMicroBatch(input, targets)
                val update = requireNotNull(second.optimizerUpdate)
                assertThat(update.updateNumber).isEqualTo(1)
                assertThat(update.microBatches).isEqualTo(2)
                assertThat(update.clipped).isTrue()
                assertThat(update.gradientNormAfterClip).isEqualTo(1e-5f)
                assertThat(trainer.optimizerUpdates).isEqualTo(1)
                assertThat(trainer.tokensSeen).isEqualTo(8)
                assertThat(model.embedding.weightParameter.array.toFloatArray()).isNotEqualTo(originalWeight)
            }
        }
    }

    @Test
    fun `flush rescales a partial accumulation and performs one update`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))
                val trainer = LanguageModelTrainer(
                    model,
                    manager,
                    tinyTrainingConfig(accumulationSteps = 3, warmupSteps = 0),
                    totalUpdates = 2,
                )
                val input = manager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                val targets = manager.create(longArrayOf(2, 3, 4, 5), Shape(1, 4))

                trainer.trainMicroBatch(input, targets)
                val update = requireNotNull(trainer.finishAccumulation())

                assertThat(update.microBatches).isEqualTo(1)
                assertThat(update.updateNumber).isEqualTo(1)
                assertThat(trainer.finishAccumulation()).isNull()
            }
        }
    }

    @Test
    fun `tiny model clearly overfits one repeated batch`() {
        Engine.getInstance().setRandomSeed(7)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))
                val updates = 60
                val trainer = LanguageModelTrainer(model, manager, tinyTrainingConfig(), updates)
                var initialLoss = Float.NaN
                var finalLoss = Float.NaN

                repeat(updates) {
                    manager.newSubManager().use { batchManager ->
                        val input = batchManager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                        val targets = batchManager.create(longArrayOf(2, 3, 4, 5), Shape(1, 4))
                        val metrics = trainer.trainMicroBatch(input, targets)
                        if (it == 0) initialLoss = metrics.loss
                        finalLoss = metrics.loss
                    }
                }

                assertThat(initialLoss).isFinite()
                assertThat(finalLoss).isFinite()
                assertThat(finalLoss).isLessThan(initialLoss * 0.25f)
                assertThat(trainer.optimizerUpdates).isEqualTo(updates)
            }
        }
    }

    @Test
    fun `restores progress and continues at the next scheduled update`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyModelConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(1, 4))
                val trainer = LanguageModelTrainer(
                    model,
                    manager,
                    tinyTrainingConfig(warmupSteps = 1),
                    totalUpdates = 6,
                    initialProgress = TrainingProgress(optimizerUpdates = 3, tokensSeen = 12),
                )
                val input = manager.create(longArrayOf(1, 2, 3, 4), Shape(1, 4))
                val targets = manager.create(longArrayOf(2, 3, 4, 5), Shape(1, 4))

                val update = requireNotNull(trainer.trainMicroBatch(input, targets).optimizerUpdate)

                assertThat(update.updateNumber).isEqualTo(4)
                assertThat(update.learningRate).isEqualTo(trainer.scheduler.learningRateForUpdate(4))
                assertThat(update.tokensSeen).isEqualTo(16)
            }
        }
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

    private fun tinyTrainingConfig(
        accumulationSteps: Int = 1,
        gradientClipNorm: Double = 1.0,
        warmupSteps: Int = 5,
    ) = TrainingConfig(
        batchSize = 1,
        gradientAccumulationSteps = accumulationSteps,
        learningRate = 0.01,
        minLearningRate = 0.001,
        warmupSteps = warmupSteps,
        weightDecay = 0.0,
        beta1 = 0.9,
        beta2 = 0.95,
        gradientClipNorm = gradientClipNorm,
        seed = 42,
    )
}
