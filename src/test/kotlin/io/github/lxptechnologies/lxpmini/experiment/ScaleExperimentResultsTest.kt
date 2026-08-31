package io.github.lxptechnologies.lxpmini.experiment

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class ScaleExperimentResultsTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `writes a stable JSON contract and a readable Markdown report`() {
        val matrix = ScaleMatrixLoader().load(Path.of("configs/pr13/matrix.yaml"))
        val result = ScaleVariantResult(
            name = "baseline-14m",
            dimension = "baseline",
            parameters = 14_266_752,
            contextLength = 16,
            dModel = 384,
            numLayers = 8,
            numHeads = 6,
            checkpointBytes = 57_075_034,
            fp32ParameterBytes = 57_067_008,
            fp32TrainingStateLowerBoundBytes = 228_268_032,
            peakJvmHeapBytes = 50_000_000,
            peakJvmHeapDeltaBytes = 25_000_000,
            elapsedSeconds = 2.5,
            tokensSeen = 96,
            tokensPerSecond = 65.7,
            trainLoss = 4.89,
            validationLoss = 5.30,
            validationPerplexity = 200.0,
            sample = "Linauuu",
        )

        ScaleResultCollector().write(temporaryDirectory, matrix, listOf(result))

        val json = Files.readString(temporaryDirectory.resolve(ScaleResultCollector.JSON_FILE))
        val markdown = Files.readString(temporaryDirectory.resolve(ScaleResultCollector.MARKDOWN_FILE))
        assertThat(json).contains("\"dModel\" : 384").doesNotContain("\"dmodel\"")
        assertThat(markdown).contains("| baseline-14m | baseline |", "`baseline-14m`: Linauuu")
    }

    @Test
    fun `heap sampler rejects a non-positive period and records its baseline`() {
        assertThatThrownBy { JvmHeapPeakSampler(0) }
            .isInstanceOf(ScaleExperimentException::class.java)
            .hasMessageContaining("must be positive")

        JvmHeapPeakSampler(1).use { sampler ->
            assertThat(sampler.peakUsedBytes).isGreaterThanOrEqualTo(sampler.baselineUsedBytes)
            assertThat(sampler.peakDeltaBytes).isGreaterThanOrEqualTo(0)
        }
    }
}
