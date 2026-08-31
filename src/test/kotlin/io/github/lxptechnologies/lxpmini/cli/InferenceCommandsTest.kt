package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizerArtifactStore
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.BeforeEach
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
@ResourceLock(value = "DJL_ENGINE", mode = ResourceAccessMode.READ_WRITE)
class InferenceCommandsTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    private lateinit var runDirectory: Path
    private lateinit var tokenizerPath: Path

    @BeforeEach
    fun createArtifacts() {
        runDirectory = temporaryDirectory.resolve("run")
        tokenizerPath = temporaryDirectory.resolve("tokenizer.json")
        val command = CheckpointDemoCommand().apply {
            configPath = Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath()
            this.runDirectory = this@InferenceCommandsTest.runDirectory
            beforeUpdates = 5
            afterUpdates = 1
        }
        check(command.call() == 0)
        ByteTokenizerArtifactStore().save(ByteTokenizer(), tokenizerPath)
    }

    @Test
    fun `complete sends repeated requests through one runtime`() {
        val captured = captureStandardOutput {
            CommandLine(LxpMiniCommand()).execute(
                "inference",
                "complete",
                "--run-dir",
                runDirectory.toString(),
                "--tokenizer",
                tokenizerPath.toString(),
                "--prompt",
                "abc",
                "--requests",
                "3",
                "--max-new-tokens",
                "1",
            )
        }

        assertThat(captured.exitCode).isZero()
        assertThat(captured.text).contains(
            "Model ID:             lxp-mini-1xm-base",
            "Model kind:           base",
            "Concurrency:          serialized",
            "Loaded once:          true",
            "Request 3:",
            "Completed requests:   3",
            "Managed arrays stable: true",
            "Runtime closed:       true",
        )
    }

    @Test
    fun `benchmark compares reload and reuse lifecycles`() {
        val captured = captureStandardOutput {
            CommandLine(LxpMiniCommand()).execute(
                "inference",
                "benchmark",
                "--run-dir",
                runDirectory.toString(),
                "--tokenizer",
                tokenizerPath.toString(),
                "--prompt",
                "abc",
                "--requests",
                "2",
                "--max-new-tokens",
                "1",
            )
        }

        assertThat(captured.exitCode).isZero()
        assertThat(captured.text).contains(
            "Requests:                  2",
            "Legacy-style model loads:  2",
            "Reused-runtime model loads: 1",
            "Outputs identical:         true",
            "Managed arrays stable:     true",
            "Runtime closed:            true",
        )
    }

    private fun captureStandardOutput(action: () -> Int): CapturedInferenceOutput {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        return try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CapturedInferenceOutput(action(), output.toString(StandardCharsets.UTF_8))
        } finally {
            System.setOut(originalOut)
        }
    }
}

private data class CapturedInferenceOutput(val exitCode: Int, val text: String)
