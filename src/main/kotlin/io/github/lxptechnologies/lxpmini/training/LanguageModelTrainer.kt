package io.github.lxptechnologies.lxpmini.training

import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.nn.Parameter
import ai.djl.training.ParameterStore
import ai.djl.training.optimizer.Optimizer
import io.github.lxptechnologies.lxpmini.config.TrainingConfig
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import kotlin.math.sqrt

class LanguageModelTrainer(
    private val model: DecoderLanguageModel,
    manager: NDManager,
    private val config: TrainingConfig,
    totalUpdates: Int,
    private val lossFunction: NextTokenCrossEntropy = NextTokenCrossEntropy(),
) {
    val scheduler = WarmupCosineScheduler(
        config.learningRate.toFloat(),
        config.minLearningRate.toFloat(),
        config.warmupSteps,
        totalUpdates,
    )
    private val parameterStore = ParameterStore(manager, false)
    private val parameters: List<Parameter> = model.parameters.values()
    private val optimizer: Optimizer = Optimizer.adamW()
        .optLearningRateTracker(scheduler)
        .optBeta1(config.beta1.toFloat())
        .optBeta2(config.beta2.toFloat())
        .optWeightDecays(config.weightDecay.toFloat())
        .build()
    private var accumulatedMicroBatches = 0

    var optimizerUpdates: Int = 0
        private set
    var tokensSeen: Long = 0
        private set

    init {
        if (!model.isInitialized) throw TrainingException("Model must be initialized before creating its trainer")
        if (config.gradientAccumulationSteps <= 0) {
            throw TrainingException("gradientAccumulationSteps must be positive")
        }
        if (!config.gradientClipNorm.isFinite() || config.gradientClipNorm <= 0.0) {
            throw TrainingException("gradientClipNorm must be finite and positive")
        }
    }

    fun trainMicroBatch(inputIds: NDArray, targetIds: NDArray): MicroBatchMetrics {
        requireBatchShapes(inputIds, targetIds)
        val loss: NDArray
        val lossValue: Float
        Engine.getInstance().newGradientCollector().use { collector ->
            val logits = model.forward(parameterStore, NDList(inputIds), true).singletonOrThrow()
            loss = lossFunction.evaluate(targetIds, logits)
            lossValue = loss.getFloat()
            if (!lossValue.isFinite()) throw TrainingException("Loss is not finite: $lossValue")
            collector.backward(loss.div(config.gradientAccumulationSteps))
        }

        accumulatedMicroBatches += 1
        tokensSeen += inputIds.size()
        val update = if (accumulatedMicroBatches == config.gradientAccumulationSteps) {
            applyOptimizerUpdate(accumulatedMicroBatches)
        } else {
            null
        }
        return MicroBatchMetrics(lossValue, accumulatedMicroBatches, update)
    }

    fun finishAccumulation(): OptimizerUpdateMetrics? =
        if (accumulatedMicroBatches == 0) null else applyOptimizerUpdate(accumulatedMicroBatches)

    private fun applyOptimizerUpdate(effectiveMicroBatches: Int): OptimizerUpdateMetrics {
        if (effectiveMicroBatches < config.gradientAccumulationSteps) {
            val correction = config.gradientAccumulationSteps.toFloat() / effectiveMicroBatches
            gradients().forEach { it.muli(correction) }
        }
        val gradientNormBeforeClip = globalGradientNorm()
        if (!gradientNormBeforeClip.isFinite()) {
            zeroGradients()
            accumulatedMicroBatches = 0
            throw TrainingException("Gradient norm is not finite: $gradientNormBeforeClip")
        }
        val clipThreshold = config.gradientClipNorm.toFloat()
        val clipped = gradientNormBeforeClip > clipThreshold
        if (clipped) {
            val scale = clipThreshold / gradientNormBeforeClip
            gradients().forEach { it.muli(scale) }
        }
        val gradientNormAfterClip = if (clipped) clipThreshold else gradientNormBeforeClip
        val nextUpdate = optimizerUpdates + 1
        val learningRate = scheduler.learningRateForUpdate(nextUpdate)
        parameters.forEach { parameter ->
            val array = parameter.array
            if (array.hasGradient()) optimizer.update(parameter.id, array, array.gradient)
        }
        zeroGradients()
        optimizerUpdates = nextUpdate
        accumulatedMicroBatches = 0
        return OptimizerUpdateMetrics(
            updateNumber = optimizerUpdates,
            learningRate = learningRate,
            gradientNormBeforeClip = gradientNormBeforeClip,
            gradientNormAfterClip = gradientNormAfterClip,
            clipped = clipped,
            microBatches = effectiveMicroBatches,
            tokensSeen = tokensSeen,
        )
    }

    private fun globalGradientNorm(): Float {
        val sumOfSquares = gradients().sumOf { gradient ->
            gradient.square().use { squared ->
                squared.sum().use { sum -> sum.getFloat().toDouble() }
            }
        }
        return sqrt(sumOfSquares).toFloat()
    }

    private fun gradients(): Sequence<NDArray> = parameters.asSequence()
        .map { it.array }
        .filter { it.hasGradient() }
        .map { it.gradient }

    private fun zeroGradients() {
        gradients().forEach { it.muli(0f) }
    }

    private fun requireBatchShapes(inputIds: NDArray, targetIds: NDArray) {
        if (inputIds.shape.dimension() != EXPECTED_RANK || targetIds.shape != inputIds.shape) {
            throw TrainingException("Inputs and targets must share shape [B, T], got ${inputIds.shape} and ${targetIds.shape}")
        }
        if (inputIds.shape[0] <= 0 || inputIds.shape[1] <= 0) {
            throw TrainingException("Batch and sequence dimensions must be positive")
        }
        if (inputIds.device != targetIds.device) {
            throw TrainingException("Inputs and targets must be on the same device")
        }
    }

    private companion object {
        const val EXPECTED_RANK = 2
    }
}

data class MicroBatchMetrics(
    val loss: Float,
    val accumulatedMicroBatches: Int,
    val optimizerUpdate: OptimizerUpdateMetrics?,
)

data class OptimizerUpdateMetrics(
    val updateNumber: Int,
    val learningRate: Float,
    val gradientNormBeforeClip: Float,
    val gradientNormAfterClip: Float,
    val clipped: Boolean,
    val microBatches: Int,
    val tokensSeen: Long,
)
