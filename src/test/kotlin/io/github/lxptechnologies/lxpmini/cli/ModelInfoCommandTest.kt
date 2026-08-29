package io.github.lxptechnologies.lxpmini.cli

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class ModelInfoCommandTest {
    @Test
    fun `model info is an executable learning experiment`() {
        val standardOutput = ByteArrayOutputStream()
        val originalOutput = System.out

        try {
            System.setOut(PrintStream(standardOutput, true, Charsets.UTF_8))
            val exitCode = CommandLine(LxpMiniCommand()).execute(
                "model",
                "info",
                "--config",
                "configs/mini-17m.yaml",
            )

            assertThat(exitCode).isZero()
            assertThat(standardOutput.toString(Charsets.UTF_8)).contains(
                "Head dimension:       64",
                "Total parameters:     17,308,032",
            )
        } finally {
            System.setOut(originalOutput)
        }
    }
}
