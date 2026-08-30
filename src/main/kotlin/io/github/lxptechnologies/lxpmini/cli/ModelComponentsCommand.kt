package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.model.RmsNorm
import io.github.lxptechnologies.lxpmini.model.RotaryPositionEmbedding
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.model.TokenEmbedding
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "components",
    mixinStandardHelpOptions = true,
    description = ["Inspect embeddings, RMSNorm, RoPE, gradients and native resource scopes."],
)
class ModelComponentsCommand : Callable<Int> {
    @Option(names = ["--vocab-size"], defaultValue = "32")
    var vocabularySize: Int = 32

    @Option(names = ["--d-model"], defaultValue = "8")
    var modelDimension: Int = 8

    @Option(names = ["--num-heads"], defaultValue = "2")
    var headCount: Int = 2

    @Option(names = ["--batch-size"], defaultValue = "2")
    var batchSize: Int = 2

    @Option(names = ["--sequence-length"], defaultValue = "4")
    var sequenceLength: Int = 4

    @Option(names = ["--context-length"], defaultValue = "16")
    var contextLength: Int = 16

    @Option(names = ["--rope-theta"], defaultValue = "10000.0")
    var ropeTheta: Double = 10_000.0

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Int = 42

    override fun call(): Int = try {
        validateOptions()
        runInspection()
        0
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun runInspection() {
        val engine = Engine.getInstance()
        engine.setRandomSeed(seed)
        val manager = NDManager.newBaseManager(Device.cpu())
        val embedding = TokenEmbedding(vocabularySize, modelDimension)
        val rmsNorm = RmsNorm(modelDimension)
        val headDimension = modelDimension / headCount

        try {
            val tokenShape = Shape(batchSize.toLong(), sequenceLength.toLong())
            embedding.initialize(manager, DataType.FLOAT32, tokenShape)
            rmsNorm.initialize(manager, DataType.FLOAT32, tokenShape.add(modelDimension.toLong()))
            val parameterStore = ParameterStore(manager, false)
            val tokenIds = manager.create(
                LongArray(batchSize * sequenceLength) { index -> (index % vocabularySize).toLong() },
                tokenShape,
            )

            RotaryPositionEmbedding(manager, headDimension, contextLength, ropeTheta).use { rope ->
                val embedded: ai.djl.ndarray.NDArray
                val normalized: ai.djl.ndarray.NDArray
                val headed: ai.djl.ndarray.NDArray
                val rotated: ai.djl.ndarray.NDArray
                engine.newGradientCollector().use { collector ->
                    embedded = embedding.forward(parameterStore, NDList(tokenIds), true).singletonOrThrow()
                    normalized = rmsNorm.forward(parameterStore, NDList(embedded), true).singletonOrThrow()
                    headed = normalized
                        .reshape(batchSize.toLong(), sequenceLength.toLong(), headCount.toLong(), headDimension.toLong())
                        .transpose(0, 2, 1, 3)
                    rotated = rope.apply(headed)
                    collector.backward(rotated.sum())
                }

                val demonstration = controlledRopeInput(manager, headDimension)
                val demonstratedRotation = rope.apply(demonstration)
                println("DJL engine:          ${engine.engineName} ${engine.version}")
                println("Device:              ${manager.device}")
                println("Token IDs shape:     ${tokenIds.shape} = [B, T]")
                println("Embedding shape:     ${embedded.shape} = [B, T, C]")
                println("RMSNorm shape:       ${normalized.shape} = [B, T, C]")
                println("Heads shape:         ${headed.shape} = [B, H, T, D]")
                println("RoPE output shape:   ${rotated.shape} = [B, H, T, D]")
                println("Head dimension:      $headDimension")
                println("RoPE cache shape:    ${rope.cacheShape()} = [context, D/2]")
                println("RoPE parameters:     0")
                println("Embedding weights:   ${vocabularySize.toLong() * modelDimension}")
                println("RMSNorm weights:     $modelDimension")
                println("Position 0 before:   ${demonstration.get("0, 0, 0, :").toFloatArray().display()}")
                println("Position 0 after:    ${demonstratedRotation.get("0, 0, 0, :").toFloatArray().display()}")
                if (sequenceLength > 1) {
                    println("Position 1 before:   ${demonstration.get("0, 0, 1, :").toFloatArray().display()}")
                    println("Position 1 after:    ${demonstratedRotation.get("0, 0, 1, :").toFloatArray().display()}")
                }
                println("Embedding grad norm: ${embedding.weightParameter.array.gradient.norm().getFloat().format()}")
                println("RMS scale grad norm: ${rmsNorm.scaleParameter.array.gradient.norm().getFloat().format()}")
                println("RoPE cache open:     ${rope.isOpen()}")
            }
        } finally {
            rmsNorm.clear()
            embedding.clear()
            manager.close()
        }
        println("Manager closed:      ${!manager.isOpen}")
    }

    private fun controlledRopeInput(manager: NDManager, headDimension: Int): ai.djl.ndarray.NDArray {
        val values = FloatArray(sequenceLength * headDimension)
        for (position in 0 until sequenceLength) {
            for (dimension in 0 until headDimension step 2) {
                values[position * headDimension + dimension] = 1f
            }
        }
        return manager.create(values, Shape(1, 1, sequenceLength.toLong(), headDimension.toLong()))
    }

    private fun validateOptions() {
        if (vocabularySize <= 0) throw TensorShapeException("--vocab-size must be positive")
        if (modelDimension <= 0) throw TensorShapeException("--d-model must be positive")
        if (headCount <= 0) throw TensorShapeException("--num-heads must be positive")
        if (modelDimension % headCount != 0) {
            throw TensorShapeException("--d-model must be divisible by --num-heads")
        }
        if ((modelDimension / headCount) % 2 != 0) {
            throw TensorShapeException("head dimension must be even for RoPE")
        }
        if (batchSize <= 0) throw TensorShapeException("--batch-size must be positive")
        if (sequenceLength <= 0) throw TensorShapeException("--sequence-length must be positive")
        if (batchSize.toLong() * sequenceLength > Int.MAX_VALUE) {
            throw TensorShapeException("--batch-size × --sequence-length exceeds the JVM array limit")
        }
        if (contextLength < sequenceLength) {
            throw TensorShapeException("--context-length must be at least --sequence-length")
        }
        if (!ropeTheta.isFinite() || ropeTheta <= 0.0) {
            throw TensorShapeException("--rope-theta must be finite and positive")
        }
    }

    private fun FloatArray.display(): String = joinToString(prefix = "[", postfix = "]") { value ->
        value.format()
    }

    private fun Float.format(): String = String.format(Locale.ROOT, "%.6f", this)
}
