package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.training.LanguageModelTrainer
import io.github.lxptechnologies.lxpmini.training.OptimizerUpdateMetrics
import io.github.lxptechnologies.lxpmini.training.TrainingException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "overfit-batch",
    mixinStandardHelpOptions = true,
    description = ["Repeat one synthetic next-token batch and verify that a tiny model memorizes it."],
)
class OverfitBatchCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
) : Callable<Int> {
    @Option(names = ["--config"], required = true, paramLabel = "<file>")
    lateinit var configPath: Path

    @Option(names = ["--updates"], defaultValue = "80")
    var updates: Int = 80

    @Option(names = ["--report-every"], defaultValue = "10")
    var reportEvery: Int = 10

    override fun call(): Int = try {
        runExperiment()
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    } catch (exception: TrainingException) {
        System.err.println("Training error: ${exception.message}")
        2
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun runExperiment() {
        val config = configLoader.load(configPath)
        validateOptions(config.model.contextLength, config.training.warmupSteps)
        val engine = Engine.getInstance()
        val engineSeed = config.training.seed.toInt()
        if (engineSeed.toLong() != config.training.seed) {
            throw TrainingException("training.seed must fit in a 32-bit integer for DJL")
        }
        engine.setRandomSeed(engineSeed)
        val manager = NDManager.newBaseManager(Device.cpu())
        val model = DecoderLanguageModel(manager, config.model)
        var initialLoss = Float.NaN
        var finalLoss = Float.NaN
        try {
            val shape = Shape(config.training.batchSize.toLong(), config.model.contextLength.toLong())
            model.initialize(manager, DataType.FLOAT32, shape)
            val trainer = LanguageModelTrainer(model, manager, config.training, updates)

            println("DJL engine:       ${engine.engineName} ${engine.version}")
            println("Device:           ${manager.device}")
            println("Batch shape:      $shape = [B, T]")
            println("Optimizer:        AdamW")
            println("Accumulation:     ${config.training.gradientAccumulationSteps} micro-batch(es)/update")

            repeat(updates) { updateIndex ->
                var updateMetrics: OptimizerUpdateMetrics? = null
                repeat(config.training.gradientAccumulationSteps) {
                    manager.newSubManager().use { batchManager ->
                        val inputs = LongArray(shape.size().toInt()) { index -> 3L + index % 4 }
                        val targets = LongArray(shape.size().toInt()) { index -> 3L + (index + 1) % 4 }
                        val metrics = trainer.trainMicroBatch(
                            batchManager.create(inputs, shape),
                            batchManager.create(targets, shape),
                        )
                        if (!initialLoss.isFinite()) initialLoss = metrics.loss
                        finalLoss = metrics.loss
                        updateMetrics = metrics.optimizerUpdate ?: updateMetrics
                    }
                }
                val completed = requireNotNull(updateMetrics)
                val updateNumber = updateIndex + 1
                if (updateNumber == 1 || updateNumber % reportEvery == 0 || updateNumber == updates) {
                    println(completed.format(finalLoss))
                }
            }
        } finally {
            model.close()
            manager.close()
        }

        println("Initial loss:     ${initialLoss.decimal()}")
        println("Final loss:       ${finalLoss.decimal()}")
        println("Reduction factor: ${(initialLoss / finalLoss).decimal()}x")
        println("Loss decreased:   ${finalLoss < initialLoss}")
        println("Manager closed:   ${!manager.isOpen}")
    }

    private fun validateOptions(contextLength: Int, warmupSteps: Int) {
        if (updates <= 0) throw TrainingException("--updates must be positive")
        if (reportEvery <= 0) throw TrainingException("--report-every must be positive")
        if (updates <= warmupSteps) {
            throw TrainingException("--updates must be greater than training.warmupSteps ($warmupSteps)")
        }
        if (contextLength <= 0) throw TensorShapeException("model.contextLength must be positive")
    }

    private fun OptimizerUpdateMetrics.format(loss: Float): String =
        "update=%4d loss=%s lr=%s grad=%s->%s clipped=%-5s tokens=%d".format(
            Locale.ROOT,
            updateNumber,
            loss.decimal(),
            learningRate.decimal(),
            gradientNormBeforeClip.decimal(),
            gradientNormAfterClip.decimal(),
            clipped,
            tokensSeen,
        )

    private fun Float.decimal(): String = String.format(Locale.ROOT, "%.6f", this)
}
