package io.github.lxptechnologies.lxpmini.checkpoint

import com.fasterxml.jackson.databind.ObjectMapper
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class RunStoreTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    private val store = RunStore()

    @Test
    fun `creates a versioned run and appends one JSON object per metric line`() {
        val config = temporaryDirectory.resolve("source.yaml")
        Files.writeString(config, "model: tiny\n")
        val runDirectory = temporaryDirectory.resolve("run")
        val initialized = store.initialize(
            runDirectory,
            config,
            RunEnvironment("PyTorch", "2.7.1", "cpu()"),
            "c".repeat(64),
            42,
        )

        store.appendMetric(runDirectory, TrainingMetricRecord("train", 1, 16, 2.5f, 0.01f, 0.5f, false))

        assertThat(initialized.configSha256).isEqualTo(Sha256.of(runDirectory.resolve("config.yaml")))
        assertThat(runDirectory.resolve("checkpoints")).isDirectory()
        assertThat(runDirectory.resolve("samples")).isDirectory()
        val metricLines = Files.readAllLines(runDirectory.resolve("metrics.jsonl"))
        assertThat(metricLines).hasSize(1)
        assertThat(ObjectMapper().readTree(metricLines.single()).get("update").asInt()).isEqualTo(1)
        assertThat(Files.readString(runDirectory.resolve("run-metadata.json")))
            .contains("\"exactTrainingResume\" : false", "\"datasetSha256\" : \"${"c".repeat(64)}\"")
    }

    @Test
    fun `refuses to mix a new run with an existing directory`() {
        val runDirectory = temporaryDirectory.resolve("run")
        Files.createDirectories(runDirectory)
        Files.writeString(runDirectory.resolve("existing.txt"), "keep")
        val config = temporaryDirectory.resolve("source.yaml")
        Files.writeString(config, "model: tiny\n")

        assertThatThrownBy {
            store.initialize(
                runDirectory,
                config,
                RunEnvironment("PyTorch", "2.7.1", "cpu()"),
                "d".repeat(64),
                42,
            )
        }.isInstanceOf(CheckpointException::class.java)
            .hasMessageContaining("absent or empty")
        assertThat(Files.readString(runDirectory.resolve("existing.txt"))).isEqualTo("keep")
    }
}
