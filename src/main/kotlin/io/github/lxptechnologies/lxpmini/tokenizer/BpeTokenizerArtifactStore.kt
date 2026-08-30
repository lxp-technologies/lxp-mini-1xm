package io.github.lxptechnologies.lxpmini.tokenizer

import com.fasterxml.jackson.annotation.JsonInclude
import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.module.kotlin.KotlinModule
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

class BpeTokenizerArtifactStore(
    private val mapper: ObjectMapper = defaultMapper(),
) {
    fun save(trainedTokenizer: TrainedBpeTokenizer, path: Path) {
        path.parent?.let { parent -> Files.createDirectories(parent) }
        val tokenizer = trainedTokenizer.tokenizer
        val artifact = BpeTokenizerArtifact(
            vocabularySize = tokenizer.vocabularySize,
            specialTokens = linkedMapOf(
                SpecialToken.PAD.tokenText to SpecialToken.PAD.id,
                SpecialToken.BOS.tokenText to SpecialToken.BOS.id,
                SpecialToken.EOS.tokenText to SpecialToken.EOS.id,
            ),
            vocabulary = (0 until tokenizer.vocabularySize).map { tokenId ->
                BpeVocabularyEntry(
                    id = tokenId,
                    bytes = tokenizer.bytesForToken(tokenId).map { byte -> byte.toInt() and UNSIGNED_BYTE_MASK },
                    specialToken = SpecialToken.fromId(tokenId)?.tokenText,
                )
            },
            merges = tokenizer.merges.map { merge ->
                BpeMergeArtifact(
                    leftId = merge.pair.leftId,
                    rightId = merge.pair.rightId,
                    resultId = merge.resultId,
                    trainingFrequency = merge.trainingFrequency,
                )
            },
            metadata = trainedTokenizer.metadata,
        )

        try {
            Files.newBufferedWriter(path).use { writer -> mapper.writeValue(writer, artifact) }
        } catch (exception: IOException) {
            throw TokenizerException("Cannot write BPE tokenizer to $path: ${exception.message}", exception)
        }
    }

    fun load(path: Path): TrainedBpeTokenizer {
        if (!Files.isRegularFile(path)) {
            throw TokenizerException("Tokenizer file does not exist: $path")
        }

        val artifact = try {
            Files.newBufferedReader(path).use { reader ->
                mapper.readValue(reader, BpeTokenizerArtifact::class.java)
            }
        } catch (exception: JsonProcessingException) {
            throw TokenizerException("Invalid tokenizer JSON in $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw TokenizerException("Cannot read tokenizer $path: ${exception.message}", exception)
        }

        validateArtifact(artifact)
        val vocabulary = artifact.vocabulary.associate { entry ->
            entry.id to entry.bytes.map(Int::toByte).toByteArray()
        }
        val merges = artifact.merges.map { merge ->
            BpeMerge(
                pair = TokenPair(merge.leftId, merge.rightId),
                resultId = merge.resultId,
                trainingFrequency = merge.trainingFrequency,
            )
        }

        return try {
            TrainedBpeTokenizer(
                tokenizer = BpeTokenizer(vocabulary, merges),
                metadata = artifact.metadata,
            )
        } catch (exception: TokenizerException) {
            throw TokenizerException("Incompatible BPE tokenizer artifact: ${exception.message}", exception)
        }
    }

    private fun validateArtifact(artifact: BpeTokenizerArtifact) {
        val expectedSpecialTokens = mapOf(
            SpecialToken.PAD.tokenText to SpecialToken.PAD.id,
            SpecialToken.BOS.tokenText to SpecialToken.BOS.id,
            SpecialToken.EOS.tokenText to SpecialToken.EOS.id,
        )
        val errors = buildList {
            if (artifact.version != FORMAT_VERSION) add("version must be $FORMAT_VERSION")
            if (artifact.type != TOKENIZER_TYPE) add("type must be '$TOKENIZER_TYPE'")
            if (artifact.byteTokenOffset != ByteTokenizer.BYTE_TOKEN_OFFSET) {
                add("byteTokenOffset must be ${ByteTokenizer.BYTE_TOKEN_OFFSET}")
            }
            if (artifact.vocabularySize < ByteTokenizer.VOCABULARY_SIZE) {
                add("vocabularySize must be at least ${ByteTokenizer.VOCABULARY_SIZE}")
            }
            if (artifact.vocabulary.size != artifact.vocabularySize) {
                add("vocabulary must contain exactly ${artifact.vocabularySize} entries")
            }
            if (artifact.vocabulary.map(BpeVocabularyEntry::id).distinct().size != artifact.vocabulary.size) {
                add("vocabulary IDs must be unique")
            }
            if (artifact.specialTokens != expectedSpecialTokens) {
                add("specialTokens must be $expectedSpecialTokens")
            }
            if (artifact.merges.size != artifact.vocabularySize - ByteTokenizer.VOCABULARY_SIZE) {
                add("merge count must equal vocabularySize - ${ByteTokenizer.VOCABULARY_SIZE}")
            }
            if (artifact.metadata.actualVocabularySize != artifact.vocabularySize) {
                add("metadata.actualVocabularySize must equal vocabularySize")
            }
            if (artifact.metadata.requestedVocabularySize < artifact.metadata.actualVocabularySize) {
                add("metadata.requestedVocabularySize cannot be smaller than actualVocabularySize")
            }
            if (artifact.metadata.corpusByteCount < 0) add("metadata.corpusByteCount must be non-negative")
            if (!artifact.metadata.corpusSha256.matches(SHA_256_PATTERN)) {
                add("metadata.corpusSha256 must be 64 lowercase hexadecimal characters")
            }
        }
        if (errors.isNotEmpty()) incompatible(errors)

        val entriesById = artifact.vocabulary.associateBy(BpeVocabularyEntry::id)
        val vocabularyErrors = buildList {
            SpecialToken.entries.forEach { specialToken ->
                val entry = entriesById[specialToken.id]
                if (entry?.specialToken != specialToken.tokenText || entry.bytes.isNotEmpty()) {
                    add("token ${specialToken.id} must represent ${specialToken.tokenText}")
                }
            }
            for (byteValue in 0..255) {
                val tokenId = byteValue + ByteTokenizer.BYTE_TOKEN_OFFSET
                val entry = entriesById[tokenId]
                if (entry?.bytes != listOf(byteValue) || entry.specialToken != null) {
                    add("token $tokenId must represent byte $byteValue")
                    break
                }
            }
            artifact.vocabulary.forEach { entry ->
                if (entry.bytes.any { byte -> byte !in 0..255 }) {
                    add("token ${entry.id} contains a byte outside 0..255")
                }
                if (entry.id !in 0 until artifact.vocabularySize) {
                    add("token ID ${entry.id} is outside the vocabulary")
                }
            }
            artifact.merges.forEachIndexed { index, merge ->
                val expectedResultId = ByteTokenizer.VOCABULARY_SIZE + index
                if (merge.resultId != expectedResultId) {
                    add("merge $index must create token ID $expectedResultId")
                }
                if (merge.trainingFrequency <= 0) {
                    add("merge $index trainingFrequency must be positive")
                }
            }
        }
        if (vocabularyErrors.isNotEmpty()) incompatible(vocabularyErrors)
    }

    private fun incompatible(errors: List<String>): Nothing {
        throw TokenizerException("Incompatible BPE tokenizer artifact: ${errors.joinToString("; ")}")
    }

    companion object {
        const val FORMAT_VERSION = 1
        const val TOKENIZER_TYPE = "byte-bpe"

        private const val UNSIGNED_BYTE_MASK = 0xFF
        private val SHA_256_PATTERN = Regex("[0-9a-f]{64}")

        private fun defaultMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
            .setDefaultPropertyInclusion(JsonInclude.Include.NON_NULL)
            .enable(SerializationFeature.INDENT_OUTPUT)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
    }
}

@JsonPropertyOrder(
    "version",
    "type",
    "vocabularySize",
    "byteTokenOffset",
    "specialTokens",
    "vocabulary",
    "merges",
    "metadata",
)
data class BpeTokenizerArtifact(
    val version: Int = BpeTokenizerArtifactStore.FORMAT_VERSION,
    val type: String = BpeTokenizerArtifactStore.TOKENIZER_TYPE,
    val vocabularySize: Int,
    val byteTokenOffset: Int = ByteTokenizer.BYTE_TOKEN_OFFSET,
    val specialTokens: Map<String, Int>,
    val vocabulary: List<BpeVocabularyEntry>,
    val merges: List<BpeMergeArtifact>,
    val metadata: BpeTrainingMetadata,
)

@JsonPropertyOrder("id", "bytes", "specialToken")
data class BpeVocabularyEntry(
    val id: Int,
    val bytes: List<Int>,
    val specialToken: String? = null,
)

@JsonPropertyOrder("leftId", "rightId", "resultId", "trainingFrequency")
data class BpeMergeArtifact(
    val leftId: Int,
    val rightId: Int,
    val resultId: Int,
    val trainingFrequency: Int,
)
