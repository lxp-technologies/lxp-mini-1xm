package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class ScaleMatrixCommandTest {
    @Test
    fun `dry run validates and prints every controlled PR13 variant without creating a model`() {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        val exitCode = try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CommandLine(LxpMiniCommand()).execute(
                "experiment",
                "scale",
                "--matrix",
                "configs/pr13/matrix.yaml",
                "--dry-run",
            )
        } finally {
            System.setOut(originalOut)
        }
        val text = output.toString(StandardCharsets.UTF_8)

        assertThat(exitCode).isZero()
        assertThat(text).contains(
            "Baseline:           baseline-14m",
            "Variants selected:  6/6",
            "width-11m",
            "11233600",
            "depth-21m",
            "21347712",
            "context-32",
            "Dry run:            true (no model instantiated)",
        )
    }
}
