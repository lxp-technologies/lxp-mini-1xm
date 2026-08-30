package io.github.lxptechnologies.lxpmini.evaluation

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.data.TokenBatch
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.training.NextTokenCrossEntropy
import kotlin.math.exp

class LanguageModelEvaluator(
    private val model: DecoderLanguageModel,
    private val manager: NDManager,
    private val lossFunction: NextTokenCrossEntropy = NextTokenCrossEntropy(),
    private val nanoTime: () -> Long = System::nanoTime,
) {
    private val parameterStore = ParameterStore(manager, false)

    init {
        if (!model.isInitialized) throw EvaluationException("Model must be initialized before evaluation")
    }

    fun evaluate(batches: Sequence<TokenBatch>, maxBatches: Int = Int.MAX_VALUE): EvaluationMetrics {
        if (maxBatches <= 0) throw EvaluationException("maxBatches must be positive")
        val startedAt = nanoTime()
        var weightedLoss = 0.0
        var tokenCount = 0L
        var batchCount = 0

        for (batch in batches.take(maxBatches)) {
            manager.newSubManager().use { temporary ->
                val shape = Shape(batch.batchSize.toLong(), batch.sequenceLength.toLong())
                val inputs = temporary.create(batch.inputIds.toLongArray(), shape)
                val targets = temporary.create(batch.targetIds.toLongArray(), shape)
                val logits = model.forward(parameterStore, NDList(inputs), false).singletonOrThrow()
                val loss = lossFunction.evaluate(targets, logits).use { value -> value.getFloat().toDouble() }
                if (!loss.isFinite()) throw EvaluationException("Evaluation loss is not finite: $loss")
                val batchTokens = batch.inputIds.size.toLong()
                weightedLoss += loss * batchTokens
                tokenCount += batchTokens
                batchCount += 1
            }
        }
        if (batchCount == 0) throw EvaluationException("Evaluation dataset contains no complete batch")

        val elapsedNanos = (nanoTime() - startedAt).coerceAtLeast(1L)
        val meanLoss = weightedLoss / tokenCount
        return EvaluationMetrics(
            loss = meanLoss,
            perplexity = perplexity(meanLoss),
            batchCount = batchCount,
            tokenCount = tokenCount,
            elapsedSeconds = elapsedNanos / NANOS_PER_SECOND,
            tokensPerSecond = tokenCount * NANOS_PER_SECOND / elapsedNanos,
        )
    }

    companion object {
        private const val NANOS_PER_SECOND = 1_000_000_000.0

        fun perplexity(loss: Double): Double {
            if (!loss.isFinite() || loss < 0.0) throw EvaluationException("loss must be finite and non-negative")
            return exp(loss)
        }
    }
}

data class EvaluationMetrics(
    val loss: Double,
    val perplexity: Double,
    val batchCount: Int,
    val tokenCount: Long,
    val elapsedSeconds: Double,
    val tokensPerSecond: Double,
)

private fun IntArray.toLongArray(): LongArray = LongArray(size) { index -> this[index].toLong() }
