package io.github.lxptechnologies.lxpmini.inference

import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.model.DecoderKeyValueCache
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import kotlin.math.max

internal interface InferenceLogitsSession : AutoCloseable {
    fun lastTokenLogits(context: IntArray): FloatArray
    fun metrics(generatedTokenCount: Int, promptTokensDiscarded: Int): InferenceMetrics
}

internal class FullRecomputeLogitsSession(
    private val requestManager: NDManager,
    private val model: DecoderLanguageModel,
    private val parameterStore: ParameterStore,
    private val contextPolicy: ContextOverflowPolicy,
) : InferenceLogitsSession {
    private var callCount = 0
    private var prefillTokens = 0L
    private var decodeTokens = 0L
    private var prefillNanos = 0L
    private var decodeNanos = 0L

    override fun lastTokenLogits(context: IntArray): FloatArray {
        val started = System.nanoTime()
        val logits = forward(context)
        val elapsed = System.nanoTime() - started
        if (callCount++ == 0) {
            prefillTokens += context.size
            prefillNanos += elapsed
        } else {
            decodeTokens += context.size
            decodeNanos += elapsed
        }
        return logits
    }

    override fun metrics(generatedTokenCount: Int, promptTokensDiscarded: Int) = InferenceMetrics(
        cacheEnabled = false,
        contextPolicy = contextPolicy,
        prefillTokensProcessed = prefillTokens,
        decodeTokensProcessed = decodeTokens,
        prefillNanos = prefillNanos,
        decodeNanos = decodeNanos,
        cacheInvalidations = 0,
        peakCachedTokens = 0,
        generatedTokenCount = generatedTokenCount,
        promptTokensDiscarded = promptTokensDiscarded,
    )

    override fun close() = Unit

    private fun forward(context: IntArray): FloatArray = requestManager.newSubManager().use { temporary ->
        val input = temporary.create(context.map(Int::toLong).toLongArray(), Shape(1, context.size.toLong()))
        val logits = model.forward(parameterStore, NDList(input), false).singletonOrThrow()
        logits.get("0, ${context.lastIndex}, :").toFloatArray()
    }
}

internal class KeyValueLogitsSession(
    private val requestManager: NDManager,
    private val model: DecoderLanguageModel,
    private val parameterStore: ParameterStore,
    private val contextPolicy: ContextOverflowPolicy,
) : InferenceLogitsSession {
    private val cache: DecoderKeyValueCache = model.newKeyValueCache(requestManager)
    private var cachedTokens = IntArray(0)
    private var prefillTokens = 0L
    private var decodeTokens = 0L
    private var prefillNanos = 0L
    private var decodeNanos = 0L
    private var invalidations = 0
    private var peakCachedTokens = 0

    override fun lastTokenLogits(context: IntArray): FloatArray {
        if (canDecodeOne(context)) return decode(context.last())
        if (cachedTokens.isNotEmpty()) {
            if (contextPolicy == ContextOverflowPolicy.REJECT) {
                throw InferenceException("Incremental context no longer extends the active KV cache")
            }
            cache.clear()
            cachedTokens = IntArray(0)
            invalidations++
        }
        return prefill(context)
    }

    override fun metrics(generatedTokenCount: Int, promptTokensDiscarded: Int) = InferenceMetrics(
        cacheEnabled = true,
        contextPolicy = contextPolicy,
        prefillTokensProcessed = prefillTokens,
        decodeTokensProcessed = decodeTokens,
        prefillNanos = prefillNanos,
        decodeNanos = decodeNanos,
        cacheInvalidations = invalidations,
        peakCachedTokens = peakCachedTokens,
        generatedTokenCount = generatedTokenCount,
        promptTokensDiscarded = promptTokensDiscarded,
    )

    override fun close() = cache.close()

    private fun prefill(context: IntArray): FloatArray {
        val started = System.nanoTime()
        val logits = forward(context)
        prefillNanos += System.nanoTime() - started
        prefillTokens += context.size
        cachedTokens = context.copyOf()
        peakCachedTokens = max(peakCachedTokens, cache.tokenCount)
        return logits
    }

    private fun decode(token: Int): FloatArray {
        val started = System.nanoTime()
        val logits = forward(intArrayOf(token))
        decodeNanos += System.nanoTime() - started
        decodeTokens++
        cachedTokens += token
        peakCachedTokens = max(peakCachedTokens, cache.tokenCount)
        return logits
    }

    private fun forward(tokens: IntArray): FloatArray = requestManager.newSubManager().use { temporary ->
        val input = temporary.create(tokens.map(Int::toLong).toLongArray(), Shape(1, tokens.size.toLong()))
        val logits = model.forwardIncremental(parameterStore, input, cache)
        logits.get("0, ${tokens.lastIndex}, :").toFloatArray()
    }

    private fun canDecodeOne(context: IntArray): Boolean {
        if (cachedTokens.isEmpty() || context.size != cachedTokens.size + 1) return false
        return cachedTokens.indices.all { index -> cachedTokens[index] == context[index] }
    }
}
