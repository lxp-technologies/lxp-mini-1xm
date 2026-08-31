package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.experiment.HeapMeasurement
import io.github.lxptechnologies.lxpmini.experiment.JvmHeapPeakSampler
import io.github.lxptechnologies.lxpmini.experiment.ScaleExperimentException
import io.github.lxptechnologies.lxpmini.experiment.ScaleMatrixLoader
import io.github.lxptechnologies.lxpmini.experiment.ScaleResultCollector
import io.github.lxptechnologies.lxpmini.experiment.ScaleVariant
import io.github.lxptechnologies.lxpmini.experiment.ValidatedScaleMatrix
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "experiment",
    description = ["Run controlled, reproducible experiment matrices."],
    subcommands = [ScaleMatrixCommand::class],
)
class ExperimentCommand : Runnable {
    override fun run() {
        println("Choose an experiment command. Try: lxp-mini experiment scale --help")
    }
}

@Command(
    name = "scale",
    mixinStandardHelpOptions = true,
    description = ["Validate or execute a one-axis-at-a-time model scale matrix."],
)
class ScaleMatrixCommand(
    private val matrixLoader: ScaleMatrixLoader = ScaleMatrixLoader(),
    private val resultCollector: ScaleResultCollector = ScaleResultCollector(),
) : Callable<Int> {
    @Option(names = ["--matrix"], required = true, paramLabel = "<file>")
    lateinit var matrixPath: Path

    @Option(names = ["--tokenizer"], paramLabel = "<file>")
    var tokenizerPath: Path? = null

    @Option(names = ["--train-corpus"], paramLabel = "<file>")
    var trainCorpusPath: Path? = null

    @Option(names = ["--validation-corpus"], paramLabel = "<file>")
    var validationCorpusPath: Path? = null

    @Option(names = ["--output-dir"], paramLabel = "<directory>")
    var outputDirectory: Path? = null

    @Option(names = ["--updates"], defaultValue = "6")
    var updates: Int = 6

    @Option(names = ["--max-validation-batches"], defaultValue = "2")
    var maxValidationBatches: Int = 2

    @Option(names = ["--shuffle-buffer"], defaultValue = "32")
    var shuffleBufferSize: Int = 32

    @Option(names = ["--prompt"], defaultValue = "Lina")
    lateinit var prompt: String

    @Option(names = ["--sample-tokens"], defaultValue = "8")
    var sampleTokens: Int = 8

    @Option(names = ["--variant"], paramLabel = "<name>", description = ["Repeat to run a subset."])
    var selectedVariantNames: Array<String> = emptyArray()

    @Option(names = ["--dry-run"], description = ["Validate and print the matrix without training."])
    var dryRun: Boolean = false

    override fun call(): Int = try {
        executeMatrix()
        0
    } catch (exception: ScaleExperimentException) {
        fail("Scale experiment", exception)
    } catch (exception: ConfigException) {
        fail("Configuration", exception)
    } catch (exception: IOException) {
        fail("I/O", exception)
    }

    private fun executeMatrix() {
        validateOptions()
        val matrix = matrixLoader.load(matrixPath)
        val variants = selectVariants(matrix)
        printPlan(matrix, variants)
        if (dryRun) {
            println("Dry run:            true (no model instantiated)")
            return
        }

        val output = requireNotNull(outputDirectory)
        requireEmptyOutputDirectory(output)
        Files.createDirectories(output)
        val results = mutableListOf<io.github.lxptechnologies.lxpmini.experiment.ScaleVariantResult>()
        for ((index, variant) in variants.withIndex()) {
            println()
            println(
                "=== Variant ${index + 1}/${variants.size}: ${variant.name} " +
                    "(${variant.dimension.name.lowercase(Locale.ROOT)}) ===",
            )
            val runDirectory = output.resolve("runs").resolve(variant.name)
            val sampler = JvmHeapPeakSampler()
            val startedAt = System.nanoTime()
            val exitCode = try {
                corpusCommand(variant, runDirectory).call()
            } finally {
                sampler.close()
            }
            if (exitCode != 0) throw ScaleExperimentException("Variant '${variant.name}' failed with exit code $exitCode")
            val elapsedSeconds = (System.nanoTime() - startedAt) / NANOS_PER_SECOND
            results += resultCollector.collect(
                variant,
                runDirectory,
                updates,
                HeapMeasurement(sampler.peakUsedBytes, sampler.peakDeltaBytes),
                elapsedSeconds,
            )
            resultCollector.write(output, matrix, results)
        }
        println()
        println("Scale JSON:         ${output.resolve(ScaleResultCollector.JSON_FILE).toAbsolutePath()}")
        println("Scale Markdown:     ${output.resolve(ScaleResultCollector.MARKDOWN_FILE).toAbsolutePath()}")
        println("Variants complete:  ${results.size}")
    }

    private fun corpusCommand(variant: ScaleVariant, runDirectory: Path): CorpusTrainingCommand =
        CorpusTrainingCommand().also { command ->
            command.configPath = variant.configPath
            command.tokenizerPath = requireNotNull(tokenizerPath)
            command.trainCorpusPath = requireNotNull(trainCorpusPath)
            command.validationCorpusPath = requireNotNull(validationCorpusPath)
            command.runDirectory = runDirectory
            command.updates = updates
            command.evaluateEvery = updates
            command.checkpointEvery = updates
            command.shuffleBufferSize = shuffleBufferSize
            command.maxValidationBatches = maxValidationBatches
            command.prompts = arrayOf(prompt)
            command.sampleTokens = sampleTokens
        }

    private fun selectVariants(matrix: ValidatedScaleMatrix): List<ScaleVariant> {
        if (selectedVariantNames.isEmpty()) return matrix.variants
        val requested = selectedVariantNames.toSet()
        val unknown = requested - matrix.variants.map(ScaleVariant::name).toSet()
        if (unknown.isNotEmpty()) throw ScaleExperimentException("Unknown variants: ${unknown.sorted().joinToString()}")
        return matrix.variants.filter { it.name in requested }
    }

    private fun printPlan(matrix: ValidatedScaleMatrix, variants: List<ScaleVariant>) {
        println("Matrix:             ${matrix.name}")
        println("Baseline:           ${matrix.baseline}")
        println("Variants selected:  ${variants.size}/${matrix.variants.size}")
        println("name             axis       parameters context dModel layers heads headDim")
        variants.forEach { variant ->
            println(
                "%-16s %-9s %10d %7d %6d %6d %5d %7d".format(
                    Locale.ROOT,
                    variant.name,
                    variant.dimension.name.lowercase(Locale.ROOT),
                    variant.parameters.total,
                    variant.model.contextLength,
                    variant.model.dModel,
                    variant.model.numLayers,
                    variant.model.numHeads,
                    variant.model.headDim,
                ),
            )
        }
    }

    private fun validateOptions() {
        if (updates <= 1) throw ScaleExperimentException("--updates must be greater than the PR13 warmup of 1")
        if (maxValidationBatches <= 0) throw ScaleExperimentException("--max-validation-batches must be positive")
        if (shuffleBufferSize < 0) throw ScaleExperimentException("--shuffle-buffer must be non-negative")
        if (prompt.isBlank()) throw ScaleExperimentException("--prompt cannot be blank")
        if (sampleTokens <= 0) throw ScaleExperimentException("--sample-tokens must be positive")
        if (!dryRun) {
            if (tokenizerPath == null) throw ScaleExperimentException("--tokenizer is required unless --dry-run")
            if (trainCorpusPath == null) throw ScaleExperimentException("--train-corpus is required unless --dry-run")
            if (validationCorpusPath == null) {
                throw ScaleExperimentException("--validation-corpus is required unless --dry-run")
            }
            if (outputDirectory == null) throw ScaleExperimentException("--output-dir is required unless --dry-run")
        }
    }

    private fun requireEmptyOutputDirectory(path: Path) {
        if (!Files.exists(path)) return
        if (!Files.isDirectory(path)) throw ScaleExperimentException("Output path is not a directory: $path")
        if (Files.list(path).use { entries -> entries.findAny().isPresent }) {
            throw ScaleExperimentException("Output directory must be absent or empty: $path")
        }
    }

    private fun fail(kind: String, exception: Exception): Int {
        System.err.println("$kind error: ${exception.message}")
        return 2
    }

    private companion object {
        const val NANOS_PER_SECOND = 1_000_000_000.0
    }
}
