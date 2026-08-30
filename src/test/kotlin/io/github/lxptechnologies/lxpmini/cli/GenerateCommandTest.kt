package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class GenerateCommandTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `CLI loads a checkpoint and traces greedy generation through the byte tokenizer`() {
        val runDirectory = temporaryDirectory.resolve("run")
        val tokenizerPath = temporaryDirectory.resolve("tokenizer.json")
        val configPath = Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath().toString()

        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "train",
                "checkpoint-demo",
                "--config",
                configPath,
                "--run-dir",
                runDirectory.toString(),
                "--before-updates",
                "5",
                "--after-updates",
                "1",
            ),
        ).isZero()
        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "byte",
                "create",
                "--output",
                tokenizerPath.toString(),
            ),
        ).isZero()

        val captured = captureStandardOutput {
            CommandLine(LxpMiniCommand()).execute(
                "generate",
                "--run-dir",
                runDirectory.toString(),
                "--tokenizer",
                tokenizerPath.toString(),
                "--prompt",
                "abc",
                "--max-new-tokens",
                "3",
                "--strategy",
                "greedy",
                "--show-candidates",
                "3",
            )
        }

        assertThat(captured.exitCode).isZero()
        assertThat(captured.text).contains(
            "Tokenizer:          byte (259 tokens)",
            "Prompt token IDs:   [100, 101, 102]",
            "Strategy:           greedy",
            "step=01 context=[100, 101, 102]",
            "candidates=[",
            "Generated token IDs:",
            "Complete text:",
            "Manager closed:     true",
        )
    }

    private fun captureStandardOutput(action: () -> Int): CapturedGenerationOutput {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        return try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CapturedGenerationOutput(action(), output.toString(StandardCharsets.UTF_8))
        } finally {
            System.setOut(originalOut)
        }
    }
}

private data class CapturedGenerationOutput(val exitCode: Int, val text: String)
