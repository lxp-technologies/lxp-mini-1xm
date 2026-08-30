package io.github.lxptechnologies.lxpmini.cli

import com.fasterxml.jackson.databind.ObjectMapper
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import picocli.CommandLine
import java.io.ByteArrayOutputStream
import java.io.PrintStream
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

@ResourceLock(value = "SYSTEM_OUT", mode = ResourceAccessMode.READ_WRITE)
class CorpusTrainingCommandTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `CLI trains evaluates checkpoints and records a measured corpus run`() {
        val tokenizerPath = temporaryDirectory.resolve("tokenizer.json")
        val runDirectory = temporaryDirectory.resolve("run")
        val trainCorpusPath = temporaryDirectory.resolve("train.txt")
        val validationCorpusPath = temporaryDirectory.resolve("validation.txt")
        Files.copy(Path.of("docs/lab-notes/samples/pr12-train.txt"), trainCorpusPath)
        Files.copy(Path.of("docs/lab-notes/samples/pr12-validation.txt"), validationCorpusPath)
        val trainCorpus = trainCorpusPath.toString()
        val validationCorpus = validationCorpusPath.toString()
        val config = Path.of("configs/lab-pr12-tiny-corpus.yaml").toAbsolutePath().toString()

        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "bpe",
                "train",
                "--input",
                trainCorpus,
                "--vocab-size",
                "272",
                "--output",
                tokenizerPath.toString(),
            ),
        ).isZero()

        val training = captureOutput {
            CommandLine(LxpMiniCommand()).execute(
                "train",
                "corpus",
                "--config",
                config,
                "--tokenizer",
                tokenizerPath.toString(),
                "--train-corpus",
                trainCorpus,
                "--validation-corpus",
                validationCorpus,
                "--run-dir",
                runDirectory.toString(),
                "--updates",
                "6",
                "--eval-every",
                "3",
                "--checkpoint-every",
                "3",
                "--shuffle-buffer",
                "4",
                "--prompt",
                "Lina",
                "--sample-tokens",
                "2",
            )
        }

        assertThat(training.exitCode).isZero()
        assertThat(training.text).contains(
            "Total tokens:",
            "update=   1",
            "update=   3",
            "update=   6",
            "Exact resume:       false",
            "Managers closed:    true",
        )
        assertThat(Files.readAllLines(runDirectory.resolve("metrics.jsonl"))).hasSize(6)
        assertThat(runDirectory.resolve("checkpoints/step-00000003/model.params")).isRegularFile()
        assertThat(runDirectory.resolve("checkpoints/step-00000006/model.params")).isRegularFile()
        assertThat(runDirectory.resolve("samples/step-00000001.txt")).isRegularFile()
        assertThat(runDirectory.resolve("samples/step-00000003.txt")).isRegularFile()
        assertThat(runDirectory.resolve("samples/step-00000006.txt")).isRegularFile()
        assertThat(runDirectory.resolve("tokenizer.json")).hasSameBinaryContentAs(tokenizerPath)
        assertThat(Files.readString(runDirectory.resolve("experiment.json")))
            .contains("\"updates\" : 6", "\"prompts\" : [ \"Lina\" ]")
        val lastMetric = ObjectMapper().readTree(Files.readAllLines(runDirectory.resolve("metrics.jsonl")).last())
        assertThat(lastMetric.get("validationLoss").asDouble()).isPositive()
        assertThat(lastMetric.get("validationPerplexity").asDouble()).isGreaterThan(1.0)

        val evaluation = captureOutput {
            CommandLine(LxpMiniCommand()).execute(
                "evaluate",
                "--run-dir",
                runDirectory.toString(),
                "--validation-corpus",
                validationCorpus,
                "--checkpoint",
                "step-00000003",
                "--checkpoint",
                "step-00000006",
            )
        }

        assertThat(evaluation.exitCode).isZero()
        assertThat(evaluation.text).contains(
            "step-00000003",
            "step-00000006",
            "Gradients computed: false",
        )

        val leakingTokenizer = temporaryDirectory.resolve("leaking-tokenizer.json")
        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "tokenizer",
                "bpe",
                "train",
                "--input",
                validationCorpus,
                "--vocab-size",
                "272",
                "--output",
                leakingTokenizer.toString(),
            ),
        ).isZero()
        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "train",
                "corpus",
                "--config",
                config,
                "--tokenizer",
                leakingTokenizer.toString(),
                "--train-corpus",
                trainCorpus,
                "--validation-corpus",
                validationCorpus,
                "--run-dir",
                temporaryDirectory.resolve("leaking-run").toString(),
                "--updates",
                "6",
            ),
        ).isEqualTo(2)

        Files.writeString(validationCorpusPath, "corpus changed", StandardCharsets.UTF_8)
        assertThat(
            CommandLine(LxpMiniCommand()).execute(
                "evaluate",
                "--run-dir",
                runDirectory.toString(),
                "--validation-corpus",
                validationCorpus,
            ),
        ).isEqualTo(2)
    }

    private fun captureOutput(action: () -> Int): CapturedCorpusOutput {
        val originalOut = System.out
        val output = ByteArrayOutputStream()
        return try {
            System.setOut(PrintStream(output, true, StandardCharsets.UTF_8))
            CapturedCorpusOutput(action(), output.toString(StandardCharsets.UTF_8))
        } finally {
            System.setOut(originalOut)
        }
    }
}

private data class CapturedCorpusOutput(val exitCode: Int, val text: String)
