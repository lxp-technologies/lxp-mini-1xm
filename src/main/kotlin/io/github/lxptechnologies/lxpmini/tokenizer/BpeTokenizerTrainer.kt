package io.github.lxptechnologies.lxpmini.tokenizer

import java.security.MessageDigest

class BpeTokenizerTrainer {
    fun train(corpusBytes: ByteArray, targetVocabularySize: Int): TrainedBpeTokenizer {
        if (targetVocabularySize < ByteTokenizer.VOCABULARY_SIZE) {
            throw TokenizerException(
                "BPE vocabulary size must be at least ${ByteTokenizer.VOCABULARY_SIZE}, " +
                    "got $targetVocabularySize",
            )
        }

        var tokenIds = IntArray(corpusBytes.size) { index ->
            val unsignedByte = corpusBytes[index].toInt() and UNSIGNED_BYTE_MASK
            unsignedByte + ByteTokenizer.BYTE_TOKEN_OFFSET
        }
        val vocabulary = createBaseVocabulary()
        val merges = mutableListOf<BpeMerge>()

        while (vocabulary.size < targetVocabularySize) {
            val pairCounts = countAdjacentPairs(tokenIds)
            val selected = selectMostFrequentPair(pairCounts)
                ?: throw TokenizerException(
                    "Corpus cannot produce $targetVocabularySize vocabulary entries; " +
                        "training stopped at ${vocabulary.size}",
                )
            val resultId = vocabulary.size
            val mergedBytes = vocabulary.getValue(selected.pair.leftId) +
                vocabulary.getValue(selected.pair.rightId)

            merges += BpeMerge(
                pair = selected.pair,
                resultId = resultId,
                trainingFrequency = selected.frequency,
            )
            vocabulary[resultId] = mergedBytes
            tokenIds = applyMerge(tokenIds, selected.pair, resultId)
        }

        val tokenizer = BpeTokenizer(
            vocabulary = vocabulary,
            merges = merges,
        )
        return TrainedBpeTokenizer(
            tokenizer = tokenizer,
            metadata = BpeTrainingMetadata(
                requestedVocabularySize = targetVocabularySize,
                actualVocabularySize = tokenizer.vocabularySize,
                corpusByteCount = corpusBytes.size.toLong(),
                corpusSha256 = corpusBytes.sha256(),
            ),
        )
    }

    internal fun countAdjacentPairs(tokenIds: IntArray): Map<TokenPair, Int> {
        if (tokenIds.size < 2) return emptyMap()

        val counts = HashMap<TokenPair, Int>()
        for (index in 0 until tokenIds.lastIndex) {
            val pair = TokenPair(tokenIds[index], tokenIds[index + 1])
            counts[pair] = counts.getOrDefault(pair, 0) + 1
        }
        return counts
    }

    internal fun selectMostFrequentPair(pairCounts: Map<TokenPair, Int>): PairSelection? {
        var bestPair: TokenPair? = null
        var bestFrequency = 0

        pairCounts.forEach { (pair, frequency) ->
            if (
                frequency > bestFrequency ||
                (frequency == bestFrequency && (bestPair == null || pair < bestPair))
            ) {
                bestPair = pair
                bestFrequency = frequency
            }
        }

        return bestPair?.let { pair -> PairSelection(pair, bestFrequency) }
    }

    internal fun applyMerge(tokenIds: IntArray, pair: TokenPair, resultId: Int): IntArray {
        val merged = IntArray(tokenIds.size)
        var inputIndex = 0
        var outputSize = 0

        while (inputIndex < tokenIds.size) {
            val pairMatches = inputIndex < tokenIds.lastIndex &&
                tokenIds[inputIndex] == pair.leftId &&
                tokenIds[inputIndex + 1] == pair.rightId

            if (pairMatches) {
                merged[outputSize++] = resultId
                inputIndex += 2
            } else {
                merged[outputSize++] = tokenIds[inputIndex]
                inputIndex += 1
            }
        }

        return merged.copyOf(outputSize)
    }

    private fun createBaseVocabulary(): LinkedHashMap<Int, ByteArray> = linkedMapOf<Int, ByteArray>().apply {
        SpecialToken.entries.forEach { specialToken -> put(specialToken.id, byteArrayOf()) }
        for (byteValue in 0..255) {
            put(byteValue + ByteTokenizer.BYTE_TOKEN_OFFSET, byteArrayOf(byteValue.toByte()))
        }
    }

    private fun ByteArray.sha256(): String = MessageDigest.getInstance("SHA-256")
        .digest(this)
        .joinToString(separator = "") { byte -> "%02x".format(byte.toInt() and UNSIGNED_BYTE_MASK) }

    private companion object {
        const val UNSIGNED_BYTE_MASK = 0xFF
    }
}

data class PairSelection(
    val pair: TokenPair,
    val frequency: Int,
)

data class BpeTrainingMetadata(
    val requestedVocabularySize: Int,
    val actualVocabularySize: Int,
    val corpusByteCount: Long,
    val corpusSha256: String,
)

data class TrainedBpeTokenizer(
    val tokenizer: BpeTokenizer,
    val metadata: BpeTrainingMetadata,
)
