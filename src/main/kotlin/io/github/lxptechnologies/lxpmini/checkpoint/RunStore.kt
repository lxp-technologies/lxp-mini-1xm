package io.github.lxptechnologies.lxpmini.checkpoint

import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
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
        datasetKind: String = DEFAULT_DATASET_KIND,
        tokenizer: String = DEFAULT_TOKENIZER,
        tokenizerSha256: String? = null,
        validationDatasetSha256: String? = null,
    ): InitializedRun {
        if (!Files.isRegularFile(configSource)) throw CheckpointException("Configuration file does not exist: $configSource")
        if (!Sha256.isValid(datasetSha256)) throw CheckpointException("datasetSha256 must be a SHA-256 value")
        if (datasetKind.isBlank()) throw CheckpointException("datasetKind cannot be blank")
        if (tokenizer.isBlank()) throw CheckpointException("tokenizer cannot be blank")
        if (tokenizerSha256 != null && !Sha256.isValid(tokenizerSha256)) {
            throw CheckpointException("tokenizerSha256 must be a SHA-256 value")
        }
        if (validationDatasetSha256 != null && !Sha256.isValid(validationDatasetSha256)) {
            throw CheckpointException("validationDatasetSha256 must be a SHA-256 value")
        }
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
                datasetKind = datasetKind,
                tokenizer = tokenizer,
                tokenizerSha256 = tokenizerSha256,
                validationDatasetSha256 = validationDatasetSha256,
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

    fun loadMetadata(runDirectory: Path): RunMetadata {
        val path = runDirectory.resolve(METADATA_FILE)
        if (!Files.isRegularFile(path)) throw CheckpointException("Run metadata does not exist: $path")
        val metadata = try {
            Files.newBufferedReader(path).use { reader -> mapper.readValue(reader, RunMetadata::class.java) }
        } catch (exception: JsonProcessingException) {
            throw CheckpointException("Invalid run metadata $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw CheckpointException("Cannot read run metadata $path: ${exception.message}", exception)
        }
        val errors = buildList {
            if (metadata.schemaVersion != FORMAT_VERSION) add("schemaVersion must be $FORMAT_VERSION")
            if (!Sha256.isValid(metadata.configSha256)) add("configSha256 must be a SHA-256 value")
            if (!Sha256.isValid(metadata.datasetSha256)) add("datasetSha256 must be a SHA-256 value")
            if (metadata.tokenizerSha256 != null && !Sha256.isValid(metadata.tokenizerSha256)) {
                add("tokenizerSha256 must be a SHA-256 value")
            }
            if (metadata.validationDatasetSha256 != null && !Sha256.isValid(metadata.validationDatasetSha256)) {
                add("validationDatasetSha256 must be a SHA-256 value")
            }
            if (metadata.datasetKind.isBlank()) add("datasetKind cannot be blank")
            if (metadata.tokenizer.isBlank()) add("tokenizer cannot be blank")
            if (metadata.exactTrainingResume) add("format 1 cannot claim exact training resume")
        }
        if (errors.isNotEmpty()) throw CheckpointException("Incompatible run metadata: ${errors.joinToString("; ")}")
        return metadata
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
        const val EXPERIMENT_FILE = "experiment.json"
        const val SAMPLES_DIRECTORY = "samples"
        const val TOKENIZER_FILE = "tokenizer.json"
        const val DEFAULT_DATASET_KIND = "synthetic-repeated-next-token-batch"
        const val DEFAULT_TOKENIZER = "not-required-synthetic-token-ids"

        private fun runMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
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
    "validationDatasetSha256",
    "tokenizer",
    "tokenizerSha256",
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
    val datasetKind: String = RunStore.DEFAULT_DATASET_KIND,
    val datasetSha256: String,
    val validationDatasetSha256: String? = null,
    val tokenizer: String = RunStore.DEFAULT_TOKENIZER,
    val tokenizerSha256: String? = null,
    val exactTrainingResume: Boolean = false,
    val limitations: List<String> = listOf(
        "AdamW first and second moments are not exposed by DJL 0.36 and are reset on resume.",
        "Backend random-number-generator state is not serialized.",
    ),
)

@JsonPropertyOrder(
    "phase",
    "update",
    "tokensSeen",
    "loss",
    "validationLoss",
    "validationPerplexity",
    "learningRate",
    "gradientNorm",
    "clipped",
    "tokensPerSecond",
    "elapsedSeconds",
)
data class TrainingMetricRecord(
    val phase: String,
    val update: Int,
    val tokensSeen: Long,
    val loss: Float,
    val learningRate: Float,
    val gradientNorm: Float,
    val clipped: Boolean,
    val validationLoss: Double? = null,
    val validationPerplexity: Double? = null,
    val tokensPerSecond: Double? = null,
    val elapsedSeconds: Double? = null,
)

data class InitializedRun(
    val directory: Path,
    val configPath: Path,
    val configSha256: String,
    val metadata: RunMetadata,
)
