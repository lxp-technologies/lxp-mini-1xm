package io.github.lxptechnologies.lxpmini.tokenizer

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.io.path.writeText

class ByteTokenizerArtifactStoreTest {
    private val store = ByteTokenizerArtifactStore()

    @Test
    fun `saves and loads a versioned byte tokenizer`(@TempDir directory: Path) {
        val artifactPath = directory.resolve("nested/tokenizer.json")
        store.save(ByteTokenizer(), artifactPath)

        val loaded = store.load(artifactPath)
        val artifactJson = artifactPath.readText()

        assertThat(loaded.decode(loaded.encode("Allô 👋"))).isEqualTo("Allô 👋")
        assertThat(artifactJson).contains(
            "\"version\" : 1",
            "\"type\" : \"byte\"",
            "\"vocabularySize\" : 259",
            "\"byteTokenOffset\" : 3",
            "\"<pad>\" : 0",
            "\"<bos>\" : 1",
            "\"<eos>\" : 2",
        )
    }

    @Test
    fun `writes deterministic artifacts`(@TempDir directory: Path) {
        val first = directory.resolve("first.json")
        val second = directory.resolve("second.json")

        store.save(ByteTokenizer(), first)
        store.save(ByteTokenizer(), second)

        assertThat(first.readText()).isEqualTo(second.readText())
    }

    @Test
    fun `rejects unknown artifact properties`(@TempDir directory: Path) {
        val artifactPath = directory.resolve("tokenizer.json")
        store.save(ByteTokenizer(), artifactPath)
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
    fun `rejects artifacts from unsupported versions`(@TempDir directory: Path) {
        val artifactPath = directory.resolve("tokenizer.json")
        store.save(ByteTokenizer(), artifactPath)
        artifactPath.writeText(artifactPath.readText().replace("\"version\" : 1", "\"version\" : 2"))

        assertThatThrownBy { store.load(artifactPath) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("version must be 1")
    }

    @Test
    fun `explains when an artifact is missing`(@TempDir directory: Path) {
        val missing = directory.resolve("missing.json")

        assertThatThrownBy { store.load(missing) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessage("Tokenizer file does not exist: $missing")
    }
}
