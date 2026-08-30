package io.github.lxptechnologies.lxpmini.tokenizer

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class TokenizerArtifactLoaderTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    private val loader = TokenizerArtifactLoader()

    @Test
    fun `loads byte and BPE artifacts through their declared type`() {
        val bytePath = temporaryDirectory.resolve("byte.json")
        ByteTokenizerArtifactStore().save(ByteTokenizer(), bytePath)
        val bpePath = temporaryDirectory.resolve("bpe.json")
        BpeTokenizerArtifactStore().save(
            BpeTokenizerTrainer().train("abababab".toByteArray(), targetVocabularySize = 261),
            bpePath,
        )

        val byteArtifact = loader.load(bytePath)
        val bpeArtifact = loader.load(bpePath)

        assertThat(byteArtifact.type).isEqualTo("byte")
        assertThat(byteArtifact.tokenizer.vocabularySize).isEqualTo(259)
        assertThat(bpeArtifact.type).isEqualTo("byte-bpe")
        assertThat(bpeArtifact.tokenizer.vocabularySize).isEqualTo(261)
    }

    @Test
    fun `rejects an unknown artifact type`() {
        val path = temporaryDirectory.resolve("unknown.json")
        Files.writeString(path, """{"version":1,"type":"wordpiece"}""")

        assertThatThrownBy { loader.load(path) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("Unsupported tokenizer type")
    }
}
