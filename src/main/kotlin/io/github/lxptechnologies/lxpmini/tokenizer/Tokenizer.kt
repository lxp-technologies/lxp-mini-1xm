package io.github.lxptechnologies.lxpmini.tokenizer

interface Tokenizer {
    val vocabularySize: Int

    fun encode(
        text: String,
        addBos: Boolean = false,
        addEos: Boolean = false,
    ): IntArray

    fun decode(
        tokenIds: IntArray,
        skipSpecialTokens: Boolean = true,
    ): String
}

class TokenizerException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
