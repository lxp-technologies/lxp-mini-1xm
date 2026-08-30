package io.github.lxptechnologies.lxpmini.tokenizer

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.ObjectMapper
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

class TokenizerArtifactLoader(
    private val byteStore: ByteTokenizerArtifactStore = ByteTokenizerArtifactStore(),
    private val bpeStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
    private val mapper: ObjectMapper = ObjectMapper(),
) {
    fun load(path: Path): LoadedTokenizerArtifact {
        if (!Files.isRegularFile(path)) throw TokenizerException("Tokenizer file does not exist: $path")
        val type = try {
            Files.newBufferedReader(path).use { reader ->
                mapper.readTree(reader).get("type")?.asText()
                    ?: throw TokenizerException("Tokenizer artifact is missing its type: $path")
            }
        } catch (exception: JsonProcessingException) {
            throw TokenizerException("Invalid tokenizer JSON in $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw TokenizerException("Cannot read tokenizer $path: ${exception.message}", exception)
        }

        return when (type) {
            ByteTokenizerArtifactStore.TOKENIZER_TYPE ->
                LoadedTokenizerArtifact(byteStore.load(path), type)

            BpeTokenizerArtifactStore.TOKENIZER_TYPE ->
                LoadedTokenizerArtifact(bpeStore.load(path).tokenizer, type)

            else -> throw TokenizerException("Unsupported tokenizer type '$type' in $path")
        }
    }
}

data class LoadedTokenizerArtifact(
    val tokenizer: Tokenizer,
    val type: String,
)
