package io.github.lxptechnologies.lxpmini.checkpoint

import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.time.Instant

class RunStore(
    private val mapper: ObjectMapper = runMapper(),
) {
    fun initialize(
        runDirectory: Path,
        configSource: Path,
        environment: RunEnvironment,
        datasetSha256: String,
        seed: Long,
    ): InitializedRun {
        if (!Files.isRegularFile(configSource)) throw CheckpointException("Configuration file does not exist: $configSource")
        if (!Sha256.isValid(datasetSha256)) throw CheckpointException("datasetSha256 must be a SHA-256 value")
        if (runDirectoryIsNotEmpty(runDirectory)) {
            throw CheckpointException("Run directory must be absent or empty: $runDirectory")
        }

        try {
            Files.createDirectories(runDirectory)
            Files.createDirectories(runDirectory.resolve(CheckpointStore.CHECKPOINTS_DIRECTORY))
            Files.createDirectories(runDirectory.resolve(SAMPLES_DIRECTORY))
            val configPath = runDirectory.resolve(CONFIG_FILE)
            Files.copy(configSource, configPath, StandardCopyOption.REPLACE_EXISTING)
            val configSha256 = Sha256.of(configPath)
            val metadata = RunMetadata(
                createdAtUtc = Instant.now().toString(),
                engineName = environment.engineName,
                engineVersion = environment.engineVersion,
                device = environment.device,
                seed = seed,
                configSha256 = configSha256,
                datasetSha256 = datasetSha256,
            )
            Files.newBufferedWriter(runDirectory.resolve(METADATA_FILE)).use { writer ->
                mapper.writerWithDefaultPrettyPrinter().writeValue(writer, metadata)
            }
            Files.createFile(runDirectory.resolve(METRICS_FILE))
            return InitializedRun(runDirectory, configPath, configSha256, metadata)
        } catch (exception: IOException) {
            throw CheckpointException("Cannot initialize run $runDirectory: ${exception.message}", exception)
        }
    }

    fun appendMetric(runDirectory: Path, metric: TrainingMetricRecord) {
        val path = runDirectory.resolve(METRICS_FILE)
        if (!Files.isRegularFile(path)) throw CheckpointException("Metrics file does not exist: $path")
        try {
            Files.newBufferedWriter(path, StandardOpenOption.APPEND).use { writer ->
                writer.write(mapper.writeValueAsString(metric))
                writer.newLine()
            }
        } catch (exception: IOException) {
            throw CheckpointException("Cannot append metric to $path: ${exception.message}", exception)
        }
    }

    private fun runDirectoryIsNotEmpty(path: Path): Boolean {
        if (!Files.exists(path)) return false
        if (!Files.isDirectory(path)) throw CheckpointException("Run path is not a directory: $path")
        return try {
            Files.list(path).use { entries -> entries.findAny().isPresent }
        } catch (exception: IOException) {
            throw CheckpointException("Cannot inspect run directory $path: ${exception.message}", exception)
        }
    }

    companion object {
        const val FORMAT_VERSION = 1
        const val CONFIG_FILE = "config.yaml"
        const val METADATA_FILE = "run-metadata.json"
        const val METRICS_FILE = "metrics.jsonl"
        const val SAMPLES_DIRECTORY = "samples"

        private fun runMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
    }
}

data class RunEnvironment(val engineName: String, val engineVersion: String, val device: String)

@JsonPropertyOrder(
    "schemaVersion",
    "createdAtUtc",
    "engineName",
    "engineVersion",
    "device",
    "seed",
    "configSha256",
    "datasetKind",
    "datasetSha256",
    "tokenizer",
    "exactTrainingResume",
    "limitations",
)
data class RunMetadata(
    val schemaVersion: Int = RunStore.FORMAT_VERSION,
    val createdAtUtc: String,
    val engineName: String,
    val engineVersion: String,
    val device: String,
    val seed: Long,
    val configSha256: String,
    val datasetKind: String = "synthetic-repeated-next-token-batch",
    val datasetSha256: String,
    val tokenizer: String = "not-required-synthetic-token-ids",
    val exactTrainingResume: Boolean = false,
    val limitations: List<String> = listOf(
        "AdamW first and second moments are not exposed by DJL 0.36 and are reset on resume.",
        "Backend random-number-generator state is not serialized.",
    ),
)

@JsonPropertyOrder("phase", "update", "tokensSeen", "loss", "learningRate", "gradientNorm", "clipped")
data class TrainingMetricRecord(
    val phase: String,
    val update: Int,
    val tokensSeen: Long,
    val loss: Float,
    val learningRate: Float,
    val gradientNorm: Float,
    val clipped: Boolean,
)

data class InitializedRun(
    val directory: Path,
    val configPath: Path,
    val configSha256: String,
    val metadata: RunMetadata,
)
