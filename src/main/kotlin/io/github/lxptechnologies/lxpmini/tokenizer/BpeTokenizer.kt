package io.github.lxptechnologies.lxpmini.tokenizer

import java.io.ByteArrayOutputStream
import java.nio.charset.StandardCharsets

class BpeTokenizer(
    vocabulary: Map<Int, ByteArray>,
    merges: List<BpeMerge>,
) : Tokenizer {
    private val vocabulary = vocabulary.mapValues { (_, bytes) -> bytes.copyOf() }
    val merges: List<BpeMerge> = merges.toList()

    override val vocabularySize: Int = this.vocabulary.size
    val maximumTokenByteLength: Int = this.vocabulary.values.maxOf(ByteArray::size)

    init {
        validateModel()
    }

    override fun encode(text: String, addBos: Boolean, addEos: Boolean): IntArray {
        val tokenIds = encodeBytes(text.toByteArray(StandardCharsets.UTF_8))

        if (!addBos && !addEos) return tokenIds

        val withSpecialTokens = IntArray(tokenIds.size + (if (addBos) 1 else 0) + (if (addEos) 1 else 0))
        var outputIndex = 0
        if (addBos) withSpecialTokens[outputIndex++] = SpecialToken.BOS.id
        tokenIds.copyInto(withSpecialTokens, destinationOffset = outputIndex)
        outputIndex += tokenIds.size
        if (addEos) withSpecialTokens[outputIndex] = SpecialToken.EOS.id
        return withSpecialTokens
    }

    fun encodeBytes(bytes: ByteArray): IntArray {
        var tokenIds = bytes.toByteTokenIds()
        merges.forEach { merge ->
            tokenIds = applyMerge(tokenIds, merge)
        }
        return tokenIds
    }

    override fun decode(tokenIds: IntArray, skipSpecialTokens: Boolean): String {
        return Utf8Codec.decode(decodeToBytes(tokenIds, skipSpecialTokens))
    }

    override fun decodeLossy(tokenIds: IntArray, skipSpecialTokens: Boolean): String {
        return Utf8Codec.decodeLossy(decodeToBytes(tokenIds, skipSpecialTokens))
    }

    override fun decodeToBytes(tokenIds: IntArray, skipSpecialTokens: Boolean): ByteArray {
        val output = ByteArrayOutputStream()
        tokenIds.forEach { tokenId ->
            val specialToken = SpecialToken.fromId(tokenId)
            when {
                specialToken != null && skipSpecialTokens -> Unit
                specialToken != null -> throw TokenizerException(
                    "Cannot decode special token ${specialToken.tokenText} when skipSpecialTokens=false",
                )

                else -> {
                    val bytes = vocabulary[tokenId] ?: throw TokenizerException(
                        "Token ID $tokenId is outside the BPE vocabulary 0..${vocabularySize - 1}",
                    )
                    output.write(bytes)
                }
            }
        }
        return output.toByteArray()
    }

    fun bytesForToken(tokenId: Int): ByteArray {
        return vocabulary[tokenId]?.copyOf()
            ?: throw TokenizerException("Token ID $tokenId is outside the BPE vocabulary 0..${vocabularySize - 1}")
    }

    fun compressionRatio(text: String): Double {
        val byteCount = text.toByteArray(StandardCharsets.UTF_8).size
        val tokenCount = encode(text).size
        return if (tokenCount == 0) 0.0 else byteCount.toDouble() / tokenCount
    }

    private fun applyMerge(tokenIds: IntArray, merge: BpeMerge): IntArray {
        val output = IntArray(tokenIds.size)
        var inputIndex = 0
        var outputSize = 0

        while (inputIndex < tokenIds.size) {
            val matches = inputIndex < tokenIds.lastIndex &&
                tokenIds[inputIndex] == merge.pair.leftId &&
                tokenIds[inputIndex + 1] == merge.pair.rightId
            if (matches) {
                output[outputSize++] = merge.resultId
                inputIndex += 2
            } else {
                output[outputSize++] = tokenIds[inputIndex]
                inputIndex += 1
            }
        }
        return output.copyOf(outputSize)
    }

    private fun ByteArray.toByteTokenIds(): IntArray = IntArray(size) { index ->
        (this[index].toInt() and UNSIGNED_BYTE_MASK) + ByteTokenizer.BYTE_TOKEN_OFFSET
    }

    private fun validateModel() {
        val expectedIds = (0 until vocabularySize).toSet()
        if (vocabulary.keys != expectedIds) {
            throw TokenizerException("BPE vocabulary IDs must be contiguous from 0 to ${vocabularySize - 1}")
        }
        if (vocabularySize < ByteTokenizer.VOCABULARY_SIZE) {
            throw TokenizerException("BPE vocabulary must contain all byte and special tokens")
        }
        SpecialToken.entries.forEach { specialToken ->
            if (vocabulary.getValue(specialToken.id).isNotEmpty()) {
                throw TokenizerException("Special token ${specialToken.tokenText} must not represent bytes")
            }
        }
        for (byteValue in 0..255) {
            val tokenId = byteValue + ByteTokenizer.BYTE_TOKEN_OFFSET
            if (!vocabulary.getValue(tokenId).contentEquals(byteArrayOf(byteValue.toByte()))) {
                throw TokenizerException("Base token $tokenId must represent byte $byteValue")
            }
        }
        merges.forEachIndexed { index, merge ->
            val expectedResultId = ByteTokenizer.VOCABULARY_SIZE + index
            if (merge.resultId != expectedResultId) {
                throw TokenizerException("BPE merge $index must create token ID $expectedResultId")
            }
            if (merge.pair.leftId !in 0 until merge.resultId || merge.pair.rightId !in 0 until merge.resultId) {
                throw TokenizerException("BPE merge $index references a token that does not exist yet")
            }
            val expectedBytes = vocabulary.getValue(merge.pair.leftId) + vocabulary.getValue(merge.pair.rightId)
            if (!vocabulary.getValue(merge.resultId).contentEquals(expectedBytes)) {
                throw TokenizerException("BPE token ${merge.resultId} bytes do not match its merge")
            }
        }
        if (merges.size != vocabularySize - ByteTokenizer.VOCABULARY_SIZE) {
            throw TokenizerException("BPE vocabulary and merge count disagree")
        }
    }

    private companion object {
        const val UNSIGNED_BYTE_MASK = 0xFF
    }
}
