package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Path
import kotlin.io.path.writeText

class BpeTokenizerCommandsTest {
    @Test
    fun `train creates an artifact that inspect can execute`(@TempDir directory: Path) {
        val corpusPath = directory.resolve("corpus.txt")
        val artifactPath = directory.resolve("artifacts/tokenizer.json")
        corpusPath.writeText("abababab")

        val trainOutput = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "bpe",
                "train",
                "--input",
                corpusPath.toString(),
                "--vocab-size",
                "261",
                "--output",
                artifactPath.toString(),
            )
            assertThat(exitCode).isZero()
        }
        val inspectOutput = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "bpe",
                "inspect",
                "--tokenizer",
                artifactPath.toString(),
                "--text",
                "abab",
                "--show-merges",
                "2",
                "--show-vocabulary",
                "2",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(artifactPath).exists()
        assertThat(trainOutput).contains(
            "Vocabulary size:    261",
            "Learned merges:     2",
        )
        assertThat(inspectOutput).contains(
            "Token IDs:          [260]",
            "Decoded text:       \"abab\"",
            "Bytes per token:    4.000",
            "#1: (100, 101) -> 259, training frequency=4",
        )
    }

    @Test
    fun `summary inspection hides full sequences but keeps measurements`(@TempDir directory: Path) {
        val corpusPath = directory.resolve("corpus.txt")
        val artifactPath = directory.resolve("tokenizer.json")
        corpusPath.writeText("abababab")
        captureStandardOutput {
            assertThat(
                CommandLine(LxpMiniCommand()).execute(
                    "tokenizer",
                    "bpe",
                    "train",
                    "--input",
                    corpusPath.toString(),
                    "--vocab-size",
                    "261",
                    "--output",
                    artifactPath.toString(),
                ),
            ).isZero()
        }

        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "bpe",
                "inspect",
                "--tokenizer",
                artifactPath.toString(),
                "--text",
                "abab",
                "--summary-only",
                "--show-merges",
                "0",
                "--show-vocabulary",
                "0",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "Byte count:         4",
            "Token count:        1",
            "Round-trip exact:   true",
        )
        assertThat(output).doesNotContain("UTF-8 bytes:", "Token IDs:", "Pieces:")
    }

    private fun captureStandardOutput(action: () -> Unit): String {
        val standardOutput = ByteArrayOutputStream()
        val originalOutput = System.out
        return try {
            System.setOut(PrintStream(standardOutput, true, Charsets.UTF_8))
            action()
            standardOutput.toString(Charsets.UTF_8)
        } finally {
            System.setOut(originalOutput)
        }
    }
}
