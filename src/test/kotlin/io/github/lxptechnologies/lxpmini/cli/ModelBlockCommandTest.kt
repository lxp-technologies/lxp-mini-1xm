package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class ModelBlockCommandTest {
    @Test
    fun `prints a finite causal block and compares residual gradient paths`() {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))

            val exitCode = picocli.CommandLine(LxpMiniCommand()).execute(
                "model",
                "block",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--ffn-dim",
                "16",
                "--sequence-length",
                "4",
                "--context-length",
                "16",
                "--seed",
                "42",
            )

            assertThat(exitCode).isZero()
            assertThat(output.toString(StandardCharsets.UTF_8)).contains(
                "SwiGLU hidden shape:       (1, 4, 16) = [B, T, F]",
                "Block output shape:        (1, 4, 8) = [B, T, C]",
                "Block parameters:          656",
                "Output finite:             true",
                "Past output max delta:     0.000000",
                "Gradient with residuals:",
                "Gradient without residuals:",
                "Gradient paths differ:     true",
                "Manager closed:            true",
            )
        } finally {
            System.setOut(originalOut)
        }
    }
}
