package io.github.lxptechnologies.lxpmini.cli

import com.fasterxml.jackson.databind.ObjectMapper
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class CheckpointDemoCommandTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `CLI interrupts reloads exact logits and records limited resume`() {
        val runDirectory = temporaryDirectory.resolve("run")
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        val exitCode = try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            picocli.CommandLine(LxpMiniCommand()).execute(
                "train",
                "checkpoint-demo",
                "--config",
                Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath().toString(),
                "--run-dir",
                runDirectory.toString(),
                "--before-updates",
                "6",
                "--after-updates",
                "2",
            )
        } finally {
            System.setOut(originalOut)
        }
        val text = output.toString(StandardCharsets.UTF_8)

        assertThat(exitCode).isZero()
        assertThat(text).contains(
            "Interrupted at update:     6",
            "Resumed through update:    8",
            "Logits exactly identical:  true",
            "Maximum logit difference:  0.0",
            "AdamW moments restored:     false",
            "Exact training resume:      false",
            "Managers closed:            true",
        )
        assertThat(Files.readAllLines(runDirectory.resolve("metrics.jsonl"))).hasSize(8)
        assertThat(Files.readString(runDirectory.resolve("checkpoints/latest.txt"))).isEqualTo("step-00000008\n")
        val manifest = ObjectMapper().readTree(
            runDirectory.resolve("checkpoints/step-00000006/manifest.json").toFile(),
        )
        assertThat(manifest.get("optimizerMomentsRestored").asBoolean()).isFalse()
        assertThat(manifest.get("exactTrainingResume").asBoolean()).isFalse()

        val verification = captureStandardOutput {
            picocli.CommandLine(LxpMiniCommand()).execute(
                "train",
                "checkpoint-verify",
                "--run-dir",
                runDirectory.toString(),
            )
        }
        assertThat(verification.exitCode).isZero()
        assertThat(verification.text).contains(
            "Optimizer updates:          8",
            "Model initialized:          true",
            "AdamW moments restored:     false",
            "Exact training resume:      false",
            "Manager closed:             true",
        )
    }

    private fun captureStandardOutput(action: () -> Int): CapturedCheckpointOutput {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        return try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CapturedCheckpointOutput(action(), output.toString(StandardCharsets.UTF_8))
        } finally {
            System.setOut(originalOut)
        }
    }
}

private data class CapturedCheckpointOutput(val exitCode: Int, val text: String)
