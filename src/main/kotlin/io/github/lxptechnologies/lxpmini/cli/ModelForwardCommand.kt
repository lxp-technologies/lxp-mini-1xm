package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.ParameterCounter
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.text.NumberFormat
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "forward",
    mixinStandardHelpOptions = true,
    description = ["Instantiate the decoder model, run logits and verify its real parameter count."],
)
class ModelForwardCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val parameterCounter: ParameterCounter = ParameterCounter(),
) : Callable<Int> {
    @Option(names = ["--config"], required = true, paramLabel = "<file>")
    lateinit var configPath: Path

    @Option(names = ["--batch-size"], defaultValue = "1")
    var batchSize: Int = 1

    @Option(names = ["--sequence-length"], defaultValue = "4")
    var sequenceLength: Int = 4

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Int = 42

    @Option(
        names = ["--untie-embeddings"],
        description = ["Give the LM head an independent C by V weight instead of reusing token embeddings."],
    )
    var untieEmbeddings: Boolean = false

    override fun call(): Int = try {
        runForward()
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun runForward() {
        val loadedConfig = configLoader.load(configPath).model
        val config = if (untieEmbeddings) loadedConfig.copy(tieEmbeddings = false) else loadedConfig
        validateRuntimeOptions(config.contextLength, config.vocabSize)
        val engine = Engine.getInstance()
        engine.setRandomSeed(seed)
        val manager = NDManager.newBaseManager(Device.cpu())
        val model = DecoderLanguageModel(manager, config)
        try {
            val tokenShape = Shape(batchSize.toLong(), sequenceLength.toLong())
            model.initialize(manager, DataType.FLOAT32, tokenShape)
            val tokenIds = manager.create(
                LongArray(batchSize * sequenceLength) { index -> (index % config.vocabSize).toLong() },
                tokenShape,
            )
            val result = model.forwardWithIntermediates(ParameterStore(manager, false), tokenIds, false)
            val theoreticalCount = parameterCounter.count(config).total
            val actualCount = model.actualParameterCount()

            println("DJL engine:              ${engine.engineName} ${engine.version}")
            println("Device:                  ${manager.device}")
            println("Token IDs shape:         ${tokenIds.shape} = [B, T]")
            println("Embedding shape:         ${result.embeddings.shape} = [B, T, C]")
            println("Transformer blocks:      ${result.blockOutputs.size}")
            println("Final RMSNorm shape:     ${result.normalizedOutput.shape} = [B, T, C]")
            println("Logits shape:            ${result.logits.shape} = [B, T, V]")
            println("Logits finite:           ${result.logits.toFloatArray().all(Float::isFinite)}")
            println("Weight tying configured: ${config.tieEmbeddings}")
            println("Same Parameter object:   ${model.sharesEmbeddingParameter()}")
            println("Same NDArray object:     ${model.sharesEmbeddingArray()}")
            println("Parameter tensors:       ${model.parameterTensorCount()}")
            println("Actual parameters:       ${actualCount.display()}")
            println("Theoretical parameters:  ${theoreticalCount.display()}")
            println("Counts match:            ${actualCount == theoreticalCount}")
            println("RoPE caches open:        ${model.openRopeCacheCount()}")
        } finally {
            model.close()
            manager.close()
        }
        println("Manager closed:          ${!manager.isOpen}")
    }

    private fun validateRuntimeOptions(contextLength: Int, vocabularySize: Int) {
        if (batchSize <= 0) throw TensorShapeException("--batch-size must be positive")
        if (sequenceLength <= 0) throw TensorShapeException("--sequence-length must be positive")
        if (sequenceLength > contextLength) {
            throw TensorShapeException("--sequence-length must not exceed model.contextLength")
        }
        if (batchSize.toLong() * sequenceLength > Int.MAX_VALUE) {
            throw TensorShapeException("--batch-size × --sequence-length exceeds the JVM array limit")
        }
        if (vocabularySize <= 0) throw TensorShapeException("model.vocabSize must be positive")
    }

    private fun Long.display(): String = NUMBER_FORMAT.format(this)

    private companion object {
        val NUMBER_FORMAT: NumberFormat = NumberFormat.getIntegerInstance(Locale.CANADA)
    }
}
