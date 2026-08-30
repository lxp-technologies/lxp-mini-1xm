package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.model.CausalSelfAttention
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.util.Locale
import java.util.concurrent.Callable
import kotlin.math.abs

@Command(
    name = "attention",
    mixinStandardHelpOptions = true,
    description = ["Inspect causal self-attention probabilities and verify that future tokens cannot leak."],
)
class ModelAttentionCommand : Callable<Int> {
    @Option(names = ["--d-model"], defaultValue = "8")
    var modelDimension: Int = 8

    @Option(names = ["--num-heads"], defaultValue = "2")
    var headCount: Int = 2

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
        inspectAttention()
        0
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun inspectAttention() {
        val engine = Engine.getInstance()
        engine.setRandomSeed(seed)
        val manager = NDManager.newBaseManager(Device.cpu())
        val attention = CausalSelfAttention(manager, modelDimension, headCount, contextLength, ropeTheta)
        try {
            val inputShape = Shape(1, sequenceLength.toLong(), modelDimension.toLong())
            attention.initialize(manager, DataType.FLOAT32, inputShape)
            val parameterStore = ParameterStore(manager, false)
            val original = controlledInput(manager)
            val changedFuture = original.duplicate()
            changedFuture.set(NDIndex("0, ${sequenceLength - 1}, :"), 1000f)

            val originalResult = attention.forwardWithAttention(parameterStore, original, training = false)
            val changedResult = attention.forwardWithAttention(parameterStore, changedFuture, training = false)
            val pastDifference = maximumPastDifference(originalResult.output, changedResult.output)

            original.setRequiresGradient(true)
            engine.newGradientCollector().use { collector ->
                val trainingOutput = attention.forwardWithAttention(parameterStore, original, training = true).output
                collector.backward(trainingOutput.sum())
            }

            println("DJL engine:             ${engine.engineName} ${engine.version}")
            println("Device:                 ${manager.device}")
            println("Input shape:            ${original.shape} = [B, T, C]")
            println("Q/K/V shape:            (1, $headCount, $sequenceLength, ${modelDimension / headCount}) = [B, H, T, D]")
            println("Attention shape:        ${originalResult.probabilities.shape} = [B, H, T, T]")
            println("Output shape:           ${originalResult.output.shape} = [B, T, C]")
            println("Attention parameters:   ${attention.parameterCount()}")
            println("Head 0 attention:")
            printMatrix(originalResult.probabilities.get("0, 0, :, :").toFloatArray())
            println("Head 0 row sums:        ${rowSums(originalResult.probabilities).display()}")
            println("Future probability max: ${maximumFutureProbability(originalResult.probabilities).format()}")
            println("Past output max delta:  ${pastDifference.format()}")
            println("Input gradient norm:    ${original.gradient.norm().getFloat().format()}")
            println("RoPE cache open:        ${attention.isRopeCacheOpen()}")
        } finally {
            attention.close()
            manager.close()
        }
        println("Manager closed:         ${!manager.isOpen}")
    }

    private fun controlledInput(manager: NDManager): NDArray {
        val values = FloatArray(sequenceLength * modelDimension)
        for (position in 0 until sequenceLength) {
            values[position * modelDimension + position % modelDimension] = 1f
        }
        return manager.create(values, Shape(1, sequenceLength.toLong(), modelDimension.toLong()))
    }

    private fun maximumPastDifference(first: NDArray, second: NDArray): Float {
        if (sequenceLength == 1) return 0f
        val pastEnd = sequenceLength - 1
        val left = first.get("0, 0:$pastEnd, :").toFloatArray()
        val right = second.get("0, 0:$pastEnd, :").toFloatArray()
        return left.indices.maxOf { index -> abs(left[index] - right[index]) }
    }

    private fun maximumFutureProbability(probabilities: NDArray): Float {
        val values = probabilities.toFloatArray()
        var maximum = 0f
        for (head in 0 until headCount) {
            for (query in 0 until sequenceLength) {
                for (key in query + 1 until sequenceLength) {
                    val index = ((head * sequenceLength + query) * sequenceLength) + key
                    maximum = maxOf(maximum, values[index])
                }
            }
        }
        return maximum
    }

    private fun rowSums(probabilities: NDArray): FloatArray =
        probabilities.get("0, 0, :, :").sum(intArrayOf(1)).toFloatArray()

    private fun printMatrix(values: FloatArray) {
        for (row in 0 until sequenceLength) {
            val offset = row * sequenceLength
            println(values.copyOfRange(offset, offset + sequenceLength).display(prefix = "  ["))
        }
    }

    private fun validateOptions() {
        if (modelDimension <= 0) throw TensorShapeException("--d-model must be positive")
        if (headCount <= 0) throw TensorShapeException("--num-heads must be positive")
        if (modelDimension % headCount != 0) {
            throw TensorShapeException("--d-model must be divisible by --num-heads")
        }
        if ((modelDimension / headCount) % 2 != 0) {
            throw TensorShapeException("head dimension must be even for RoPE")
        }
        if (sequenceLength <= 0) throw TensorShapeException("--sequence-length must be positive")
        if (contextLength < sequenceLength) {
            throw TensorShapeException("--context-length must be at least --sequence-length")
        }
        if (sequenceLength.toLong() * sequenceLength > Int.MAX_VALUE) {
            throw TensorShapeException("attention matrix exceeds the JVM array limit")
        }
        if (!ropeTheta.isFinite() || ropeTheta <= 0.0) {
            throw TensorShapeException("--rope-theta must be finite and positive")
        }
    }

    private fun FloatArray.display(prefix: String = "["): String = joinToString(prefix = prefix, postfix = "]") {
        it.format()
    }

    private fun Float.format(): String = String.format(Locale.ROOT, "%.6f", this)
}
