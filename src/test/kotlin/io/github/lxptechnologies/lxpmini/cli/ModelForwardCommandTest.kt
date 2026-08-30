package io.github.lxptechnologies.lxpmini.cli

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
class ModelForwardCommandTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `prints finite logits real counts and object identity for tied weights`() {
        val config = writeTinyConfig()
        val output = captureStandardOutput {
            picocli.CommandLine(LxpMiniCommand()).execute(
                "model",
                "forward",
                "--config",
                config.toString(),
                "--batch-size",
                "1",
                "--sequence-length",
                "4",
                "--seed",
                "42",
            )
        }

        assertThat(output.exitCode).isZero()
        assertThat(output.text).contains(
            "Logits shape:            (1, 4, 259) = [B, T, V]",
            "Logits finite:           true",
            "Weight tying configured: true",
            "Same Parameter object:   true",
            "Same NDArray object:     true",
            "Parameter tensors:       20",
            "Actual parameters:       3,392",
            "Theoretical parameters:  3,392",
            "Counts match:            true",
            "Manager closed:          true",
        )
    }

    @Test
    fun `untie option adds a real independent output matrix`() {
        val config = writeTinyConfig()
        val output = captureStandardOutput {
            picocli.CommandLine(LxpMiniCommand()).execute(
                "model",
                "forward",
                "--config",
                config.toString(),
                "--sequence-length",
                "4",
                "--untie-embeddings",
            )
        }

        assertThat(output.exitCode).isZero()
        assertThat(output.text).contains(
            "Weight tying configured: false",
            "Same Parameter object:   false",
            "Same NDArray object:     false",
            "Parameter tensors:       21",
            "Actual parameters:       5,464",
            "Theoretical parameters:  5,464",
            "Counts match:            true",
        )
    }

    private fun writeTinyConfig(): Path {
        val path = temporaryDirectory.resolve("tiny.yaml")
        Files.writeString(
            path,
            """
            model:
              vocabSize: 259
              contextLength: 8
              dModel: 8
              numLayers: 2
              numHeads: 2
              ffnDim: 16
              ropeTheta: 10000.0
              dropout: 0.0
              tieEmbeddings: true
            training:
              batchSize: 1
              gradientAccumulationSteps: 1
              learningRate: 0.001
              minLearningRate: 0.0001
              warmupSteps: 0
              weightDecay: 0.0
              beta1: 0.9
              beta2: 0.95
              gradientClipNorm: 1.0
              seed: 42
            """.trimIndent(),
        )
        return path
    }

    private fun captureStandardOutput(action: () -> Int): CapturedOutput {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        return try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CapturedOutput(action(), output.toString(StandardCharsets.UTF_8))
        } finally {
            System.setOut(originalOut)
        }
    }
}

private data class CapturedOutput(
    val exitCode: Int,
    val text: String,
)
