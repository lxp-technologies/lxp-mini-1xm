package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class ModelComponentsCommandTest {
    @Test
    fun `components command exposes shapes rotations gradients and resource closure`() {
        val output = captureStandardOutput {
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "model",
                "components",
                "--vocab-size",
                "16",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--batch-size",
                "1",
                "--sequence-length",
                "2",
                "--context-length",
                "4",
            )
            assertThat(exitCode).isZero()
        }

        assertThat(output).contains(
            "DJL engine:          PyTorch",
            "Token IDs shape:     (1, 2) = [B, T]",
            "Embedding shape:     (1, 2, 8) = [B, T, C]",
            "Heads shape:         (1, 2, 2, 4) = [B, H, T, D]",
            "RoPE cache shape:    (4, 2) = [context, D/2]",
            "RoPE parameters:     0",
            "Position 0 before:   [1.000000, 0.000000, 1.000000, 0.000000]",
            "Position 0 after:    [1.000000, 0.000000, 1.000000, 0.000000]",
            "Manager closed:      true",
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
