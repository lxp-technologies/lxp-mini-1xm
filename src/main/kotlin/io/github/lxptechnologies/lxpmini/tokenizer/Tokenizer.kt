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

    fun decodeToBytes(
        tokenIds: IntArray,
        skipSpecialTokens: Boolean = true,
    ): ByteArray

    fun decodeLossy(
        tokenIds: IntArray,
        skipSpecialTokens: Boolean = true,
    ): String = decode(tokenIds, skipSpecialTokens)
}

class TokenizerException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
