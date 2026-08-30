package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import java.nio.file.Path

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class OverfitBatchCommandTest {
    @Test
    fun `tiny preset overfits one deterministic batch through the CLI`() {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        val exitCode = try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            picocli.CommandLine(LxpMiniCommand()).execute(
                "train",
                "overfit-batch",
                "--config",
                Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath().toString(),
                "--updates",
                "80",
                "--report-every",
                "20",
            )
        } finally {
            System.setOut(originalOut)
        }
        val text = output.toString(StandardCharsets.UTF_8)

        assertThat(exitCode).isZero()
        assertThat(text).contains(
            "Batch shape:      (2, 8) = [B, T]",
            "update=   1",
            "update=  80",
            "tokens=1280",
            "Loss decreased:   true",
            "Manager closed:   true",
        )
    }
}
