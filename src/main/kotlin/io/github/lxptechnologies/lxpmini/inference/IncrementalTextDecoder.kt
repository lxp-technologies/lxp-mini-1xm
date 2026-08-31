package io.github.lxptechnologies.lxpmini.inference

import io.github.lxptechnologies.lxpmini.tokenizer.Tokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException

internal class IncrementalTextDecoder(
    private val tokenizer: Tokenizer,
) {
    private val tokenIds = ArrayList<Int>()
    private var emittedText = ""

    fun accept(tokenId: Int, onTextDelta: (String) -> Unit) {
        tokenIds += tokenId
        val decoded = try {
            tokenizer.decode(tokenIds.toIntArray())
        } catch (_: TokenizerException) {
            return
        }
        emitRemaining(decoded, onTextDelta)
    }

    fun finish(onTextDelta: (String) -> Unit): String {
        val decoded = tokenizer.decodeLossy(tokenIds.toIntArray())
        emitRemaining(decoded, onTextDelta)
        return decoded
    }

    private fun emitRemaining(decoded: String, onTextDelta: (String) -> Unit) {
        if (!decoded.startsWith(emittedText)) {
            throw InferenceException("Tokenizer streaming output is not prefix-stable")
        }
        val delta = decoded.substring(emittedText.length)
        if (delta.isNotEmpty()) {
            onTextDelta(delta)
            emittedText = decoded
        }
    }
}
