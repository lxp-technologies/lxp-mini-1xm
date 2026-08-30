package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.model.TransformerBlock
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.util.Locale
import java.util.concurrent.Callable
import kotlin.math.abs

@Command(
    name = "block",
    mixinStandardHelpOptions = true,
    description = ["Inspect a pre-norm Transformer block, SwiGLU and residual gradient paths."],
)
class ModelBlockCommand : Callable<Int> {
    @Option(names = ["--d-model"], defaultValue = "8")
    var modelDimension: Int = 8

    @Option(names = ["--num-heads"], defaultValue = "2")
    var headCount: Int = 2

    @Option(names = ["--ffn-dim"], defaultValue = "16")
    var hiddenDimension: Int = 16

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
        inspectBlock()
        0
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun inspectBlock() {
        val engine = Engine.getInstance()
        engine.setRandomSeed(seed)
        val manager = NDManager.newBaseManager(Device.cpu())
        val block = TransformerBlock(
            manager,
            modelDimension,
            headCount,
            hiddenDimension,
            contextLength,
            ropeTheta,
        )
        try {
            val inputShape = Shape(1, sequenceLength.toLong(), modelDimension.toLong())
            block.initialize(manager, DataType.FLOAT32, inputShape)
            val parameterStore = ParameterStore(manager, false)
            val input = controlledInput(manager)
            val changedFuture = input.duplicate()
            changedFuture.set(NDIndex("0, ${sequenceLength - 1}, :"), 1000f)
            val result = block.forwardForInspection(parameterStore, input, training = false)
            val changedResult = block.forwardForInspection(parameterStore, changedFuture, training = false)
            val pastDifference = maximumPastDifference(result.output, changedResult.output)

            val residualInput = controlledInput(manager).also { it.setRequiresGradient(true) }
            val noResidualInput = controlledInput(manager).also { it.setRequiresGradient(true) }
            val residualGradientNorm = gradientNorm(block, parameterStore, residualInput, true)
            val noResidualGradientNorm = gradientNorm(block, parameterStore, noResidualInput, false)

            println("DJL engine:                ${engine.engineName} ${engine.version}")
            println("Device:                    ${manager.device}")
            println("Input shape:               ${input.shape} = [B, T, C]")
            println("Attention norm shape:      ${result.normalizedAttentionInput.shape}")
            println("Attention output shape:    ${result.attentionOutput.shape}")
            println("First residual shape:      ${result.afterAttentionResidual.shape}")
            println("Feed-forward norm shape:   ${result.normalizedFeedForwardInput.shape}")
            println("SwiGLU hidden shape:       ${result.feedForwardHidden.shape} = [B, T, F]")
            println("Feed-forward output shape: ${result.feedForwardOutput.shape}")
            println("Block output shape:        ${result.output.shape} = [B, T, C]")
            println("Attention parameters:      ${block.attention.parameterCount()}")
            println("SwiGLU parameters:         ${block.feedForward.parameterCount()}")
            println("RMSNorm parameters:        ${2L * modelDimension}")
            println("Block parameters:          ${block.parameterCount()}")
            println("Output finite:             ${result.output.toFloatArray().all(Float::isFinite)}")
            println("Past output max delta:     ${pastDifference.format()}")
            println("Gradient with residuals:   ${residualGradientNorm.format()}")
            println("Gradient without residuals: ${noResidualGradientNorm.format()}")
            println("Gradient paths differ:     ${residualGradientNorm != noResidualGradientNorm}")
            println("RoPE cache open:           ${block.isRopeCacheOpen()}")
        } finally {
            block.close()
            manager.close()
        }
        println("Manager closed:            ${!manager.isOpen}")
    }

    private fun gradientNorm(
        block: TransformerBlock,
        parameterStore: ParameterStore,
        input: NDArray,
        includeResidualConnections: Boolean,
    ): Float = Engine.getInstance().newGradientCollector().use { collector ->
        val output = block.forwardForInspection(
            parameterStore,
            input,
            training = true,
            includeResidualConnections,
        ).output
        collector.backward(output.square().mean())
        input.gradient.norm().getFloat()
    }

    private fun controlledInput(manager: NDManager): NDArray {
        val elementCount = sequenceLength * modelDimension
        val values = FloatArray(elementCount) { index -> (index + 1f) / elementCount }
        return manager.create(values, Shape(1, sequenceLength.toLong(), modelDimension.toLong()))
    }

    private fun maximumPastDifference(first: NDArray, second: NDArray): Float {
        if (sequenceLength == 1) return 0f
        val end = sequenceLength - 1
        val left = first.get("0, 0:$end, :").toFloatArray()
        val right = second.get("0, 0:$end, :").toFloatArray()
        return left.indices.maxOf { index -> abs(left[index] - right[index]) }
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
        if (hiddenDimension <= 0) throw TensorShapeException("--ffn-dim must be positive")
        if (sequenceLength <= 0) throw TensorShapeException("--sequence-length must be positive")
        if (sequenceLength.toLong() * modelDimension > Int.MAX_VALUE) {
            throw TensorShapeException("--sequence-length × --d-model exceeds the JVM array limit")
        }
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

    private fun Float.format(): String = String.format(Locale.ROOT, "%.6f", this)
}
