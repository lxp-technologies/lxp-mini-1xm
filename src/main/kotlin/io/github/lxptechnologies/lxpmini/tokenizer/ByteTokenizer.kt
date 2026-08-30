package io.github.lxptechnologies.lxpmini.tokenizer

import java.nio.charset.StandardCharsets

class ByteTokenizer : Tokenizer {
    override val vocabularySize: Int = VOCABULARY_SIZE

    override fun encode(
        text: String,
        addBos: Boolean,
        addEos: Boolean,
    ): IntArray {
        val utf8Bytes = text.toByteArray(StandardCharsets.UTF_8)
        val prefixSize = if (addBos) 1 else 0
        val suffixSize = if (addEos) 1 else 0
        val tokenIds = IntArray(prefixSize + utf8Bytes.size + suffixSize)
        var outputIndex = 0

        if (addBos) {
            tokenIds[outputIndex++] = SpecialToken.BOS.id
        }
        utf8Bytes.forEach { byte ->
            tokenIds[outputIndex++] = tokenIdForByte(byte.toInt() and UNSIGNED_BYTE_MASK)
        }
        if (addEos) {
            tokenIds[outputIndex] = SpecialToken.EOS.id
        }

        return tokenIds
    }

    override fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean): String {
        return Utf8Codec.decode(decodeToBytes(tokenIds, skipSpecialTokens))
    }

    fun decodeToBytes(tokenIds: IntArray, skipSpecialTokens: Boolean = true): ByteArray {
        val output = ByteArray(tokenIds.size)
        var outputSize = 0

        tokenIds.forEach { tokenId ->
            val specialToken = SpecialToken.fromId(tokenId)
            when {
                tokenId in FIRST_BYTE_TOKEN_ID..LAST_BYTE_TOKEN_ID -> {
                    output[outputSize++] = byteValueForTokenId(tokenId).toByte()
                }

                specialToken != null && skipSpecialTokens -> Unit
                specialToken != null -> {
                    throw TokenizerException(
                        "Cannot decode special token ${specialToken.tokenText} when skipSpecialTokens=false",
                    )
                }

                else -> throw TokenizerException(
                    "Token ID $tokenId is outside the byte tokenizer vocabulary 0..${VOCABULARY_SIZE - 1}",
                )
            }
        }

        return output.copyOf(outputSize)
    }

    fun tokenIdForByte(unsignedByte: Int): Int {
        if (unsignedByte !in MIN_BYTE_VALUE..MAX_BYTE_VALUE) {
            throw TokenizerException("Byte value must be in 0..255, got $unsignedByte")
        }
        return unsignedByte + BYTE_TOKEN_OFFSET
    }

    fun byteValueForTokenId(tokenId: Int): Int {
        if (tokenId !in FIRST_BYTE_TOKEN_ID..LAST_BYTE_TOKEN_ID) {
            throw TokenizerException(
                "Token ID $tokenId does not represent a byte; expected $FIRST_BYTE_TOKEN_ID..$LAST_BYTE_TOKEN_ID",
            )
        }
        return tokenId - BYTE_TOKEN_OFFSET
    }

    companion object {
        const val BYTE_TOKEN_OFFSET = 3
        const val VOCABULARY_SIZE = 259
        const val FIRST_BYTE_TOKEN_ID = BYTE_TOKEN_OFFSET
        const val LAST_BYTE_TOKEN_ID = BYTE_TOKEN_OFFSET + 255

        private const val MIN_BYTE_VALUE = 0
        private const val MAX_BYTE_VALUE = 255
        private const val UNSIGNED_BYTE_MASK = 0xFF
    }
}
