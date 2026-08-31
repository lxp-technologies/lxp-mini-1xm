package io.github.lxptechnologies.lxpmini.generation

class AutoregressiveGenerator(
    private val contextLength: Int,
    private val vocabularySize: Int,
    private val sampler: TokenSampler,
    private val logitsProvider: (IntArray) -> FloatArray,
) {
    init {
        if (contextLength <= 0) throw GenerationException("contextLength must be positive")
        if (vocabularySize <= 1) throw GenerationException("vocabularySize must be greater than one")
    }

    fun generate(
        promptTokenIds: IntArray,
        maxNewTokens: Int,
        eosTokenId: Int,
        options: SamplingOptions,
        onStep: (GenerationStep) -> Unit = {},
    ): GenerationResult {
        if (promptTokenIds.isEmpty()) throw GenerationException("Prompt must contain at least one token")
        if (maxNewTokens < 0) throw GenerationException("maxNewTokens must be non-negative")
        if (eosTokenId !in 0 until vocabularySize) throw GenerationException("EOS token is outside the vocabulary")
        requireTokenIds(promptTokenIds)

        val allTokens = promptTokenIds.toMutableList()
        val generated = ArrayList<Int>(maxNewTokens)
        val steps = ArrayList<GenerationStep>(maxNewTokens)
        var stoppedByEos = false

        for (stepIndex in 0 until maxNewTokens) {
            val context = allTokens.takeLast(contextLength).toIntArray()
            val logits = logitsProvider(context)
            if (logits.size != vocabularySize) {
                throw GenerationException("Logits size ${logits.size} does not match vocabularySize $vocabularySize")
            }
            val sampling = sampler.select(logits, options)
            allTokens += sampling.tokenId
            generated += sampling.tokenId
            val step = GenerationStep(stepIndex + 1, context, sampling)
            steps += step
            onStep(step)
            if (sampling.tokenId == eosTokenId) {
                stoppedByEos = true
                break
            }
        }

        return GenerationResult(promptTokenIds.copyOf(), generated.toIntArray(), stoppedByEos, steps)
    }

    private fun requireTokenIds(tokenIds: IntArray) {
        tokenIds.firstOrNull { tokenId -> tokenId !in 0 until vocabularySize }?.let { invalid ->
            throw GenerationException("Token ID $invalid is outside vocabulary 0..${vocabularySize - 1}")
        }
    }
}

data class GenerationStep(
    val number: Int,
    val contextTokenIds: IntArray,
    val sampling: SamplingResult,
)

data class GenerationResult(
    val promptTokenIds: IntArray,
    val generatedTokenIds: IntArray,
    val stoppedByEos: Boolean,
    val steps: List<GenerationStep>,
) {
    val allTokenIds: IntArray = promptTokenIds + generatedTokenIds
}
