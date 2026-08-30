package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.file.Path
import kotlin.io.path.writeText

class ByteTokenizerCommandsTest {
    @Test
    fun `inspect makes UTF-8 bytes and token IDs visible`() {
        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "byte",
                "inspect",
                "--text",
                "é👋",
                "--add-bos",
                "--add-eos",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "UTF-8 bytes:        [195, 169, 240, 159, 145, 139]",
            "Token IDs:          [1, 198, 172, 243, 162, 148, 142, 2]",
            "Decoded text:       \"é👋\"",
            "Vocabulary size:    259",
        )
    }

    @Test
    fun `create writes an executable tokenizer artifact`(@TempDir directory: Path) {
        val artifactPath = directory.resolve("artifacts/tokenizer.json")

        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "byte",
                "create",
                "--output",
                artifactPath.toString(),
            )
            assertThat(exitCode).isZero()
        }

        assertThat(artifactPath).exists()
        assertThat(output).contains("Byte tokenizer written to $artifactPath")
    }

    @Test
    fun `inspect reads Unicode from a UTF-8 file`(@TempDir directory: Path) {
        val textPath = directory.resolve("unicode.txt")
        textPath.writeText("é👋")

        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "byte",
                "inspect",
                "--text-file",
                textPath.toString(),
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "Text:               \"é👋\"",
            "UTF-8 bytes:        [195, 169, 240, 159, 145, 139]",
            "Token IDs:          [198, 172, 243, 162, 148, 142]",
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
