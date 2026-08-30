package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class ModelAttentionCommandTest {
    @Test
    fun `prints a causal four-token attention experiment`() {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))

            val exitCode = picocli.CommandLine(LxpMiniCommand()).execute(
                "model",
                "attention",
                "--d-model",
                "8",
                "--num-heads",
                "2",
                "--sequence-length",
                "4",
                "--context-length",
                "16",
                "--seed",
                "42",
            )

            assertThat(exitCode).isZero()
            assertThat(output.toString(StandardCharsets.UTF_8)).contains(
                "Attention shape:        (1, 2, 4, 4) = [B, H, T, T]",
                "Attention parameters:   256",
                "Head 0 row sums:        [1.000000, 1.000000, 1.000000, 1.000000]",
                "Future probability max: 0.000000",
                "Past output max delta:  0.000000",
                "Manager closed:         true",
            )
        } finally {
            System.setOut(originalOut)
        }
    }
}
