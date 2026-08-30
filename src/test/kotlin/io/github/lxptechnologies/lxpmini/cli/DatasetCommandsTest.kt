package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerTrainer
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import kotlin.io.path.writeText

class DatasetCommandsTest {
    @Test
    fun `window command makes the next-token shift visible`() {
        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "dataset",
                "window",
                "--tokens",
                "10,20,30,40,50",
                "--context-length",
                "4",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "input:  [10, 20, 30, 40]",
            "target: [20, 30, 40, 50]",
            "Complete windows:   1",
            "Trailing tokens:    0",
        )
    }

    @Test
    fun `inspect streams a real corpus into disjoint ranges and batches`(@TempDir directory: Path) {
        val corpusPath = directory.resolve("corpus.txt")
        val tokenizerPath = directory.resolve("tokenizer.json")
        val corpus = "abcdefghij"
        corpusPath.writeText(corpus)
        val trained = BpeTokenizerTrainer().train(
            corpus.toByteArray(StandardCharsets.UTF_8),
            targetVocabularySize = 259,
        )
        BpeTokenizerArtifactStore().save(trained, tokenizerPath)

        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "dataset",
                "inspect",
                "--corpus",
                corpusPath.toString(),
                "--tokenizer",
                tokenizerPath.toString(),
                "--context-length",
                "3",
                "--batch-size",
                "2",
                "--validation-fraction",
                "0.2",
                "--byte-chunk-size",
                "1",
                "--show-batches",
                "1",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "Total tokens:       10",
            "Train range:        [0, 8) = 8 tokens",
            "Validation range:   [8, 10) = 2 tokens",
            "Complete windows:   2",
            "Trailing tokens:    1",
            "Batch #1 actual shape [2, 3]",
            "row 0 input:  [100, 101, 102]",
            "row 0 target: [101, 102, 103]",
            "row 1 input:  [103, 104, 105]",
            "row 1 target: [104, 105, 106]",
        )
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
