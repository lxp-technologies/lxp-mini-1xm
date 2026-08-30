package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointException
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunEnvironment
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.checkpoint.TrainingMetricRecord
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.config.ProjectConfig
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.training.LanguageModelTrainer
import io.github.lxptechnologies.lxpmini.training.OptimizerUpdateMetrics
import io.github.lxptechnologies.lxpmini.training.TrainingException
import io.github.lxptechnologies.lxpmini.training.TrainingProgress
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.concurrent.Callable
import kotlin.math.abs

@Command(
    name = "checkpoint-demo",
    mixinStandardHelpOptions = true,
    description = ["Train, checkpoint, recreate the model, verify logits and continue with declared resume limits."],
)
class CheckpointDemoCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val runStore: RunStore = RunStore(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
) : Callable<Int> {
    @Option(names = ["--config"], required = true, paramLabel = "<file>")
    lateinit var configPath: Path

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--before-updates"], defaultValue = "10")
    var beforeUpdates: Int = 10

    @Option(names = ["--after-updates"], defaultValue = "5")
    var afterUpdates: Int = 5

    override fun call(): Int = try {
        runDemo()
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    } catch (exception: CheckpointException) {
        System.err.println("Checkpoint error: ${exception.message}")
        2
    } catch (exception: TrainingException) {
        System.err.println("Training error: ${exception.message}")
        2
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun runDemo() {
        val config = configLoader.load(configPath)
        val totalUpdates = beforeUpdates + afterUpdates
        validateOptions(config, totalUpdates)
        val engine = Engine.getInstance()
        val seed = exactEngineSeed(config.training.seed)
        engine.setRandomSeed(seed)
        val shape = Shape(config.training.batchSize.toLong(), config.model.contextLength.toLong())
        val batch = syntheticBatch(shape)
        val datasetSha256 = Sha256.of(batch.canonicalDescription.toByteArray(StandardCharsets.UTF_8))
        val initializedRun = runStore.initialize(
            runDirectory,
            configPath,
            RunEnvironment(engine.engineName, engine.version, Device.cpu().toString()),
            datasetSha256,
            config.training.seed,
        )

        val logitsBeforeClose: FloatArray
        val firstCheckpointSha: String
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                model.initialize(manager, DataType.FLOAT32, shape)
                val trainer = LanguageModelTrainer(model, manager, config.training, totalUpdates)
                trainUpdates("before-checkpoint", beforeUpdates, config, trainer, manager, shape, batch)
                logitsBeforeClose = logits(model, manager, shape, batch.inputs)
                val saved = checkpointStore.save(
                    runDirectory,
                    model,
                    TrainingProgress(trainer.optimizerUpdates, trainer.tokensSeen),
                    totalUpdates,
                    initializedRun.configSha256,
                )
                firstCheckpointSha = saved.manifest.modelSha256
            }
        }

        engine.setRandomSeed(seed)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                val loaded = checkpointStore.loadLatest(
                    runDirectory,
                    model,
                    manager,
                    initializedRun.configSha256,
                )
                val logitsAfterLoad = logits(model, manager, shape, batch.inputs)
                val maximumDifference = logitsBeforeClose.indices.maxOf { index ->
                    abs(logitsBeforeClose[index] - logitsAfterLoad[index])
                }
                val trainer = LanguageModelTrainer(
                    model,
                    manager,
                    config.training,
                    totalUpdates,
                    initialProgress = loaded.progress,
                )
                trainUpdates("after-resume", afterUpdates, config, trainer, manager, shape, batch)
                checkpointStore.save(
                    runDirectory,
                    model,
                    TrainingProgress(trainer.optimizerUpdates, trainer.tokensSeen),
                    totalUpdates,
                    initializedRun.configSha256,
                )

                println("Run directory:             ${runDirectory.toAbsolutePath()}")
                println("Interrupted at update:     $beforeUpdates")
                println("Resumed through update:    ${trainer.optimizerUpdates}")
                println("Tokens seen:               ${trainer.tokensSeen}")
                println("Checkpoint model SHA-256:  $firstCheckpointSha")
                println("Logits exactly identical:  ${logitsBeforeClose.contentEquals(logitsAfterLoad)}")
                println("Maximum logit difference:  $maximumDifference")
                println("Optimizer counter restored: true")
                println("Scheduler restored:         true")
                println("AdamW moments restored:     false")
                println("Random state restored:      false")
                println("Exact training resume:      false")
            }
        }
        println("Managers closed:            true")
    }

    private fun trainUpdates(
        phase: String,
        updateCount: Int,
        config: ProjectConfig,
        trainer: LanguageModelTrainer,
        manager: NDManager,
        shape: Shape,
        batch: SyntheticBatch,
    ) {
        repeat(updateCount) {
            var optimizerMetrics: OptimizerUpdateMetrics? = null
            var loss = Float.NaN
            repeat(config.training.gradientAccumulationSteps) {
                manager.newSubManager().use { batchManager ->
                    val metrics = trainer.trainMicroBatch(
                        batchManager.create(batch.inputs, shape),
                        batchManager.create(batch.targets, shape),
                    )
                    loss = metrics.loss
                    optimizerMetrics = metrics.optimizerUpdate ?: optimizerMetrics
                }
            }
            val update = requireNotNull(optimizerMetrics)
            runStore.appendMetric(
                runDirectory,
                TrainingMetricRecord(
                    phase,
                    update.updateNumber,
                    update.tokensSeen,
                    loss,
                    update.learningRate,
                    update.gradientNormBeforeClip,
                    update.clipped,
                ),
            )
        }
    }

    private fun logits(
        model: DecoderLanguageModel,
        manager: NDManager,
        shape: Shape,
        inputs: LongArray,
    ): FloatArray = manager.newSubManager().use { temporary ->
        val tokenIds = temporary.create(inputs, shape)
        model.forward(ParameterStore(temporary, false), ai.djl.ndarray.NDList(tokenIds), false)
            .singletonOrThrow()
            .toFloatArray()
    }

    private fun syntheticBatch(shape: Shape): SyntheticBatch {
        val size = shape.size().toInt()
        val pattern = "abc ".toByteArray(StandardCharsets.US_ASCII)
            .map { byte -> (byte.toInt() + ByteTokenizer.BYTE_TOKEN_OFFSET).toLong() }
        val inputs = LongArray(size) { index -> pattern[index % pattern.size] }
        val targets = LongArray(size) { index -> pattern[(index + 1) % pattern.size] }
        val description = "shape=${shape[0]}x${shape[1]};inputs=${inputs.joinToString(",")};targets=${targets.joinToString(",")}"
        return SyntheticBatch(inputs, targets, description)
    }

    private fun validateOptions(config: ProjectConfig, totalUpdates: Int) {
        if (beforeUpdates <= 0) throw TrainingException("--before-updates must be positive")
        if (afterUpdates <= 0) throw TrainingException("--after-updates must be positive")
        if (totalUpdates <= config.training.warmupSteps) {
            throw TrainingException("Total updates must be greater than training.warmupSteps (${config.training.warmupSteps})")
        }
    }

    private fun exactEngineSeed(seed: Long): Int {
        val converted = seed.toInt()
        if (converted.toLong() != seed) throw TrainingException("training.seed must fit in a 32-bit integer for DJL")
        return converted
    }
}

private data class SyntheticBatch(
    val inputs: LongArray,
    val targets: LongArray,
    val canonicalDescription: String,
)
