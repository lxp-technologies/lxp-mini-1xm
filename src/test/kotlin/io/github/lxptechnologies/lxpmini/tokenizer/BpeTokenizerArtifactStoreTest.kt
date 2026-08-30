package io.github.lxptechnologies.lxpmini.tokenizer

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.io.path.writeText

class BpeTokenizerArtifactStoreTest {
    private val trainer = BpeTokenizerTrainer()
    private val store = BpeTokenizerArtifactStore()

    @Test
    fun `saves and loads an executable BPE tokenizer`(@TempDir directory: Path) {
        val artifactPath = directory.resolve("nested/tokenizer.json")
        val trained = trainer.train("abababab".utf8Bytes(), targetVocabularySize = 261)

        store.save(trained, artifactPath)
        val loaded = store.load(artifactPath)

        assertThat(loaded.metadata).isEqualTo(trained.metadata)
        assertThat(loaded.tokenizer.merges).isEqualTo(trained.tokenizer.merges)
        assertThat(loaded.tokenizer.bytesForToken(260)).isEqualTo(trained.tokenizer.bytesForToken(260))
        assertThat(loaded.tokenizer.decode(loaded.tokenizer.encode("abab"))).isEqualTo("abab")
        assertThat(artifactPath.readText()).contains(
            "\"type\" : \"byte-bpe\"",
            "\"vocabularySize\" : 261",
            "\"resultId\" : 259",
            "\"trainingFrequency\" : 4",
            "\"corpusSha256\"",
        )
    }

    @Test
    fun `writes deterministic artifacts`(@TempDir directory: Path) {
        val trained = trainer.train("abababab".utf8Bytes(), targetVocabularySize = 261)
        val first = directory.resolve("first.json")
        val second = directory.resolve("second.json")

        store.save(trained, first)
        store.save(trained, second)

        assertThat(first.readText()).isEqualTo(second.readText())
    }

    @Test
    fun `rejects unknown artifact properties`(@TempDir directory: Path) {
        val artifactPath = saveExample(directory)
        artifactPath.writeText(
            artifactPath.readText().replace(
                "\"version\" : 1,",
                "\"version\" : 1,\n  \"hiddenMagic\" : true,",
            ),
        )

        assertThatThrownBy { store.load(artifactPath) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("hiddenMagic")
    }

    @Test
    fun `rejects vocabulary bytes that disagree with a merge`(@TempDir directory: Path) {
        val artifactPath = saveExample(directory)
        artifactPath.writeText(
            artifactPath.readText().replace(
                "\"bytes\" : [ 97, 98 ]",
                "\"bytes\" : [ 97, 99 ]",
            ),
        )

        assertThatThrownBy { store.load(artifactPath) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("bytes do not match its merge")
    }

    @Test
    fun `rejects a merge that references an unknown token`(@TempDir directory: Path) {
        val artifactPath = saveExample(directory)
        artifactPath.writeText(artifactPath.readText().replace("\"leftId\" : 100", "\"leftId\" : -1"))

        assertThatThrownBy { store.load(artifactPath) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("references a token that does not exist yet")
    }

    private fun saveExample(directory: Path): Path = directory.resolve("tokenizer.json").also { path ->
        store.save(trainer.train("abababab".utf8Bytes(), targetVocabularySize = 261), path)
    }

    private fun String.utf8Bytes(): ByteArray = toByteArray(StandardCharsets.UTF_8)
}
