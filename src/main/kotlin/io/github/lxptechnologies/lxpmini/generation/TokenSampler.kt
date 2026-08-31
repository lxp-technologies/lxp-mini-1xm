package io.github.lxptechnologies.lxpmini.generation

import java.util.Random
import kotlin.math.exp

class TokenSampler(seed: Long) {
    private val random = Random(seed)

    fun select(
        logits: FloatArray,
        options: SamplingOptions,
        allowedTokenIds: BooleanArray? = null,
    ): SamplingResult {
        validate(logits, options, allowedTokenIds)
        val selectableTokenIds = logits.indices.filter { tokenId -> allowedTokenIds?.get(tokenId) != false }
        if (options.strategy == SamplingStrategy.GREEDY) {
            val tokenId = selectableTokenIds.maxBy { index -> logits[index] }
            return SamplingResult(tokenId, 1.0, listOf(TokenProbability(tokenId, 1.0, logits[tokenId].toDouble())))
        }

        val sorted = selectableTokenIds
            .map { tokenId -> ScaledLogit(tokenId, logits[tokenId].toDouble() / options.temperature) }
            .sortedWith(compareByDescending<ScaledLogit>(ScaledLogit::value).thenBy(ScaledLogit::tokenId))
        val topKCandidates = if (options.topK == 0) sorted else sorted.take(options.topK)
        val maximum = topKCandidates.first().value
        val weights = topKCandidates.map { candidate -> exp(candidate.value - maximum) }
        val weightSum = weights.sum()
        val probabilities = topKCandidates.indices.map { index ->
            TokenProbability(
                topKCandidates[index].tokenId,
                weights[index] / weightSum,
                topKCandidates[index].value,
            )
        }

        var cumulative = 0.0
        var nucleusSize = probabilities.size
        for (index in probabilities.indices) {
            cumulative += probabilities[index].probability
            if (cumulative >= options.topP) {
                nucleusSize = index + 1
                break
            }
        }
        val nucleus = probabilities.take(nucleusSize)
        val nucleusSum = nucleus.sumOf(TokenProbability::probability)
        val normalized = nucleus.map { candidate -> candidate.copy(probability = candidate.probability / nucleusSum) }
        val draw = random.nextDouble()
        var sampled = normalized.last()
        cumulative = 0.0
        for (candidate in normalized) {
            cumulative += candidate.probability
            if (draw < cumulative) {
                sampled = candidate
                break
            }
        }
        return SamplingResult(sampled.tokenId, sampled.probability, normalized)
    }

    private fun validate(logits: FloatArray, options: SamplingOptions, allowedTokenIds: BooleanArray?) {
        if (logits.isEmpty()) throw GenerationException("Logits cannot be empty")
        if (logits.any { value -> !value.isFinite() }) throw GenerationException("All logits must be finite")
        if (allowedTokenIds != null && allowedTokenIds.size != logits.size) {
            throw GenerationException("Allowed token mask size must match logits size")
        }
        if (allowedTokenIds != null && allowedTokenIds.none { it }) {
            throw GenerationException("Allowed token mask must contain at least one token")
        }
        if (!options.temperature.isFinite() || options.temperature <= 0.0) {
            throw GenerationException("temperature must be finite and positive")
        }
        if (options.topK !in 0..logits.size) {
            throw GenerationException("topK must be 0 or in [1, ${logits.size}]")
        }
        if (!options.topP.isFinite() || options.topP <= 0.0 || options.topP > 1.0) {
            throw GenerationException("topP must be in (0.0, 1.0]")
        }
    }
}

enum class SamplingStrategy {
    GREEDY,
    SAMPLE,
}

data class SamplingOptions(
    val strategy: SamplingStrategy = SamplingStrategy.SAMPLE,
    val temperature: Double = 1.0,
    val topK: Int = 0,
    val topP: Double = 1.0,
)

data class TokenProbability(
    val tokenId: Int,
    val probability: Double,
    val scaledLogit: Double,
)

data class SamplingResult(
    val tokenId: Int,
    val probability: Double,
    val candidates: List<TokenProbability>,
)

private data class ScaledLogit(val tokenId: Int, val value: Double)
