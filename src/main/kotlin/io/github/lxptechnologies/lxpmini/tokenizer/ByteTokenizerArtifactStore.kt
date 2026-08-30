package io.github.lxptechnologies.lxpmini.tokenizer

import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.module.kotlin.KotlinModule
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

class ByteTokenizerArtifactStore(
    private val mapper: ObjectMapper = defaultMapper(),
) {
    fun save(tokenizer: ByteTokenizer, path: Path) {
        path.parent?.let { parent -> Files.createDirectories(parent) }
        val artifact = ByteTokenizerArtifact(
            vocabularySize = tokenizer.vocabularySize,
            specialTokens = linkedMapOf(
                SpecialToken.PAD.tokenText to SpecialToken.PAD.id,
                SpecialToken.BOS.tokenText to SpecialToken.BOS.id,
                SpecialToken.EOS.tokenText to SpecialToken.EOS.id,
            ),
        )

        try {
            Files.newBufferedWriter(path).use { writer -> mapper.writeValue(writer, artifact) }
        } catch (exception: IOException) {
            throw TokenizerException("Cannot write byte tokenizer to $path: ${exception.message}", exception)
        }
    }

    fun load(path: Path): ByteTokenizer {
        if (!Files.isRegularFile(path)) {
            throw TokenizerException("Tokenizer file does not exist: $path")
        }

        val artifact = try {
            Files.newBufferedReader(path).use { reader ->
                mapper.readValue(reader, ByteTokenizerArtifact::class.java)
            }
        } catch (exception: JsonProcessingException) {
            throw TokenizerException("Invalid tokenizer JSON in $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw TokenizerException("Cannot read tokenizer $path: ${exception.message}", exception)
        }

        validate(artifact)
        return ByteTokenizer()
    }

    private fun validate(artifact: ByteTokenizerArtifact) {
        val expectedSpecialTokens = mapOf(
            SpecialToken.PAD.tokenText to SpecialToken.PAD.id,
            SpecialToken.BOS.tokenText to SpecialToken.BOS.id,
            SpecialToken.EOS.tokenText to SpecialToken.EOS.id,
        )
        val errors = buildList {
            if (artifact.version != FORMAT_VERSION) add("version must be $FORMAT_VERSION")
            if (artifact.type != TOKENIZER_TYPE) add("type must be '$TOKENIZER_TYPE'")
            if (artifact.vocabularySize != ByteTokenizer.VOCABULARY_SIZE) {
                add("vocabularySize must be ${ByteTokenizer.VOCABULARY_SIZE}")
            }
            if (artifact.byteTokenOffset != ByteTokenizer.BYTE_TOKEN_OFFSET) {
                add("byteTokenOffset must be ${ByteTokenizer.BYTE_TOKEN_OFFSET}")
            }
            if (artifact.specialTokens != expectedSpecialTokens) {
                add("specialTokens must be $expectedSpecialTokens")
            }
        }

        if (errors.isNotEmpty()) {
            throw TokenizerException("Incompatible byte tokenizer artifact: ${errors.joinToString("; ")}")
        }
    }

    companion object {
        const val FORMAT_VERSION = 1
        const val TOKENIZER_TYPE = "byte"

        private fun defaultMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
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
)
data class ByteTokenizerArtifact(
    val version: Int = ByteTokenizerArtifactStore.FORMAT_VERSION,
    val type: String = ByteTokenizerArtifactStore.TOKENIZER_TYPE,
    val vocabularySize: Int,
    val byteTokenOffset: Int = ByteTokenizer.BYTE_TOKEN_OFFSET,
    val specialTokens: Map<String, Int>,
)
