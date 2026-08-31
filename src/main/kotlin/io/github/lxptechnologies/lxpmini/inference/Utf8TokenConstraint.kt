package io.github.lxptechnologies.lxpmini.inference

import io.github.lxptechnologies.lxpmini.generation.GenerationTokenConstraint
import io.github.lxptechnologies.lxpmini.tokenizer.Tokenizer

internal class Utf8TokenConstraint(
    private val tokenizer: Tokenizer,
    vocabularySize: Int,
    private val eosTokenId: Int,
) : GenerationTokenConstraint {
    private val tokenBytes = Array(vocabularySize) { tokenId ->
        tokenizer.decodeToBytes(intArrayOf(tokenId))
    }

    override fun allowedTokenIds(generatedTokenIds: IntArray, remainingSteps: Int): BooleanArray {
        val currentState = Utf8PrefixState().consume(tokenizer.decodeToBytes(generatedTokenIds))
        check(currentState.valid) { "Generated token prefix must be valid UTF-8" }

        return BooleanArray(tokenBytes.size) { tokenId ->
            if (tokenId == eosTokenId) {
                currentState.pendingContinuationBytes == 0
            } else {
                val candidateState = currentState.consume(tokenBytes[tokenId])
                candidateState.valid && candidateState.pendingContinuationBytes <= remainingSteps
            }
        }
    }
}

private data class Utf8PrefixState(
    val valid: Boolean = true,
    val pendingContinuationBytes: Int = 0,
    val nextByteMinimum: Int = CONTINUATION_MIN,
    val nextByteMaximum: Int = CONTINUATION_MAX,
) {
    fun consume(bytes: ByteArray): Utf8PrefixState {
        var state = this
        bytes.forEach { byte ->
            state = state.consume(byte.toInt() and UNSIGNED_BYTE_MASK)
            if (!state.valid) return state
        }
        return state
    }

    private fun consume(byte: Int): Utf8PrefixState {
        if (!valid) return this
        if (pendingContinuationBytes > 0) {
            if (byte !in nextByteMinimum..nextByteMaximum) return INVALID
            return copy(
                pendingContinuationBytes = pendingContinuationBytes - 1,
                nextByteMinimum = CONTINUATION_MIN,
                nextByteMaximum = CONTINUATION_MAX,
            )
        }

        return when (byte) {
            in 0x00..0x7F -> this
            in 0xC2..0xDF -> continuationState(1)
            0xE0 -> continuationState(2, firstMinimum = 0xA0)
            in 0xE1..0xEC, in 0xEE..0xEF -> continuationState(2)
            0xED -> continuationState(2, firstMaximum = 0x9F)
            0xF0 -> continuationState(3, firstMinimum = 0x90)
            in 0xF1..0xF3 -> continuationState(3)
            0xF4 -> continuationState(3, firstMaximum = 0x8F)
            else -> INVALID
        }
    }

    private fun continuationState(
        count: Int,
        firstMinimum: Int = CONTINUATION_MIN,
        firstMaximum: Int = CONTINUATION_MAX,
    ) = Utf8PrefixState(
        pendingContinuationBytes = count,
        nextByteMinimum = firstMinimum,
        nextByteMaximum = firstMaximum,
    )

    private companion object {
        const val UNSIGNED_BYTE_MASK = 0xFF
        const val CONTINUATION_MIN = 0x80
        const val CONTINUATION_MAX = 0xBF
        val INVALID = Utf8PrefixState(valid = false)

    }
}
