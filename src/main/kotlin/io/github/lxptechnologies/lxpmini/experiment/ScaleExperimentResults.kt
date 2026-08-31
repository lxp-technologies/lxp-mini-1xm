package io.github.lxptechnologies.lxpmini.experiment

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.module.kotlin.KotlinModule
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.time.Instant
import java.util.Locale

class ScaleResultCollector(
    private val mapper: ObjectMapper = resultMapper(),
) {
    fun collect(
        variant: ScaleVariant,
        runDirectory: Path,
        update: Int,
        heap: HeapMeasurement,
        elapsedSeconds: Double,
    ): ScaleVariantResult {
        val metric = readLastMetric(runDirectory)
        val checkpointId = CheckpointStore.checkpointId(update)
        val checkpointFile = runDirectory.resolve(CheckpointStore.CHECKPOINTS_DIRECTORY)
            .resolve(checkpointId).resolve(CheckpointStore.MODEL_FILE)
        if (!Files.isRegularFile(checkpointFile)) {
            throw ScaleExperimentException("Final checkpoint does not exist: $checkpointFile")
        }
        val samplePath = runDirectory.resolve(RunStore.SAMPLES_DIRECTORY).resolve("$checkpointId.txt")
        val sample = if (Files.isRegularFile(samplePath)) {
            Files.readAllLines(samplePath).firstOrNull().orEmpty().substringAfter('\t').removePrefix("text=")
        } else {
            ""
        }
        val parameters = variant.parameters.total
        return ScaleVariantResult(
            name = variant.name,
            dimension = variant.dimension.name.lowercase(Locale.ROOT),
            parameters = parameters,
            contextLength = variant.model.contextLength,
            dModel = variant.model.dModel,
            numLayers = variant.model.numLayers,
            numHeads = variant.model.numHeads,
            checkpointBytes = Files.size(checkpointFile),
            fp32ParameterBytes = Math.multiplyExact(parameters, Float.SIZE_BYTES.toLong()),
            fp32TrainingStateLowerBoundBytes = Math.multiplyExact(parameters, FP32_TRAINING_BYTES_PER_PARAMETER),
            peakJvmHeapBytes = heap.peakUsedBytes,
            peakJvmHeapDeltaBytes = heap.peakDeltaBytes,
            elapsedSeconds = elapsedSeconds,
            tokensSeen = metric.requiredLong("tokensSeen"),
            tokensPerSecond = metric.requiredDouble("tokensPerSecond"),
            trainLoss = metric.requiredDouble("loss"),
            validationLoss = metric.requiredDouble("validationLoss"),
            validationPerplexity = metric.requiredDouble("validationPerplexity"),
            sample = sample,
        )
    }

    fun write(outputDirectory: Path, matrix: ValidatedScaleMatrix, results: List<ScaleVariantResult>) {
        val report = ScaleExperimentReport(
            matrix = matrix.name,
            baseline = matrix.baseline,
            generatedAtUtc = Instant.now().toString(),
            memoryScope = "checkpoint bytes are measured; FP32 training state is a lower bound; JVM heap excludes DJL native memory",
            results = results,
        )
        try {
            Files.newBufferedWriter(outputDirectory.resolve(JSON_FILE)).use { writer -> mapper.writeValue(writer, report) }
            Files.writeString(outputDirectory.resolve(MARKDOWN_FILE), report.markdown())
        } catch (exception: IOException) {
            throw ScaleExperimentException("Cannot write scale results: ${exception.message}", exception)
        }
    }

    private fun readLastMetric(runDirectory: Path): JsonNode {
        val path = runDirectory.resolve(RunStore.METRICS_FILE)
        val line = try {
            Files.readAllLines(path).lastOrNull()
        } catch (exception: IOException) {
            throw ScaleExperimentException("Cannot read metrics $path: ${exception.message}", exception)
        } ?: throw ScaleExperimentException("Metrics file is empty: $path")
        return try {
            mapper.readTree(line)
        } catch (exception: IOException) {
            throw ScaleExperimentException("Invalid final metric in $path: ${exception.message}", exception)
        }
    }

    private fun JsonNode.requiredDouble(field: String): Double = get(field)?.takeUnless(JsonNode::isNull)?.asDouble()
        ?: throw ScaleExperimentException("Final metric is missing '$field'")

    private fun JsonNode.requiredLong(field: String): Long = get(field)?.takeUnless(JsonNode::isNull)?.asLong()
        ?: throw ScaleExperimentException("Final metric is missing '$field'")

    companion object {
        const val JSON_FILE = "scale-results.json"
        const val MARKDOWN_FILE = "scale-results.md"
        private const val FP32_TRAINING_BYTES_PER_PARAMETER = 16L

        private fun resultMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
            .enable(SerializationFeature.INDENT_OUTPUT)
    }
}

data class HeapMeasurement(val peakUsedBytes: Long, val peakDeltaBytes: Long)

@JsonPropertyOrder("matrix", "baseline", "generatedAtUtc", "memoryScope", "results")
data class ScaleExperimentReport(
    val matrix: String,
    val baseline: String,
    val generatedAtUtc: String,
    val memoryScope: String,
    val results: List<ScaleVariantResult>,
) {
    fun markdown(): String = buildString {
        appendLine("# Scale experiment: $matrix")
        appendLine()
        appendLine("Baseline: `$baseline`")
        appendLine()
        appendLine("Memory scope: $memoryScope.")
        appendLine()
        appendLine("| Variant | Axis | Params | Context | Layers | Checkpoint MiB | FP32 state lower bound MiB | Heap peak delta MiB | tokens/s | Train loss | Validation loss | Perplexity |")
        appendLine("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        results.forEach { result ->
            appendLine(
                "| ${result.name} | ${result.dimension} | ${result.parameters} | ${result.contextLength} | " +
                    "${result.numLayers} | ${result.checkpointBytes.mib()} | " +
                    "${result.fp32TrainingStateLowerBoundBytes.mib()} | ${result.peakJvmHeapDeltaBytes.mib()} | " +
                    "${result.tokensPerSecond.number()} | ${result.trainLoss.number()} | " +
                    "${result.validationLoss.number()} | ${result.validationPerplexity.number()} |",
            )
        }
        appendLine()
        appendLine("## Fixed-prompt samples")
        appendLine()
        results.forEach { result -> appendLine("- `${result.name}`: ${result.sample.replace("|", "\\|")}") }
    }

    private fun Long.mib(): String = (this / BYTES_PER_MIB).number()
    private fun Double.number(): String = "%.4f".format(Locale.ROOT, this)

    private companion object {
        const val BYTES_PER_MIB = 1024.0 * 1024.0
    }
}

@JsonPropertyOrder(
    "name",
    "dimension",
    "parameters",
    "contextLength",
    "dModel",
    "numLayers",
    "numHeads",
    "checkpointBytes",
    "fp32ParameterBytes",
    "fp32TrainingStateLowerBoundBytes",
    "peakJvmHeapBytes",
    "peakJvmHeapDeltaBytes",
    "elapsedSeconds",
    "tokensSeen",
    "tokensPerSecond",
    "trainLoss",
    "validationLoss",
    "validationPerplexity",
    "sample",
)
data class ScaleVariantResult(
    val name: String,
    val dimension: String,
    val parameters: Long,
    val contextLength: Int,
    @get:JsonProperty("dModel")
    val dModel: Int,
    val numLayers: Int,
    val numHeads: Int,
    val checkpointBytes: Long,
    val fp32ParameterBytes: Long,
    val fp32TrainingStateLowerBoundBytes: Long,
    val peakJvmHeapBytes: Long,
    val peakJvmHeapDeltaBytes: Long,
    val elapsedSeconds: Double,
    val tokensSeen: Long,
    val tokensPerSecond: Double,
    val trainLoss: Double,
    val validationLoss: Double,
    val validationPerplexity: Double,
    val sample: String,
)
