package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.generation.GenerationException
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.inference.CompletionRequest
import io.github.lxptechnologies.lxpmini.inference.ContextOverflowPolicy
import io.github.lxptechnologies.lxpmini.inference.InferenceException
import io.github.lxptechnologies.lxpmini.inference.InferenceRuntimeLoader
import io.github.lxptechnologies.lxpmini.inference.TokenGenerationRequest
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerArtifactLoader
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable
import kotlin.math.max

@Command(
    name = "inference",
    description = ["Serve repeated local inference requests from one loaded base model."],
    subcommands = [
        InferenceCompleteCommand::class,
        InferenceBenchmarkCommand::class,
        InferenceCacheBenchmarkCommand::class,
    ],
)
class InferenceCommand : Runnable {
    override fun run() {
        println("Choose an inference command. Try: lxp-mini inference complete --help")
    }
}

@Command(
    name = "complete",
    mixinStandardHelpOptions = true,
    description = ["Load the runtime once and complete one or more prompts."],
)
class InferenceCompleteCommand(
    private val runtimeLoader: InferenceRuntimeLoader = InferenceRuntimeLoader(),
) : Callable<Int> {
    @Option(names = ["--model-id"], defaultValue = DEFAULT_MODEL_ID)
    lateinit var modelId: String

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--prompt"], required = true, paramLabel = "<text>")
    lateinit var prompt: String

    @Option(names = ["--requests"], defaultValue = "1", description = ["Number of requests sent to the same runtime."])
    var requests: Int = 1

    @Option(names = ["--max-new-tokens"], defaultValue = "16")
    var maxNewTokens: Int = 16

    @Option(names = ["--strategy"], defaultValue = "greedy", description = ["greedy or sample"])
    lateinit var strategyName: String

    @Option(names = ["--temperature"], defaultValue = "1.0")
    var temperature: Double = 1.0

    @Option(names = ["--top-k"], defaultValue = "0")
    var topK: Int = 0

    @Option(names = ["--top-p"], defaultValue = "1.0")
    var topP: Double = 1.0

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Long = 42

    @Option(names = ["--add-bos"])
    var addBos: Boolean = false

    @Option(names = ["--kv-cache"], negatable = true, defaultValue = "true")
    var cacheEnabled: Boolean = true

    @Option(names = ["--context-policy"], defaultValue = "sliding-window", description = ["sliding-window or reject"])
    lateinit var contextPolicyName: String

    override fun call(): Int = inferenceCli {
        requirePositiveRequests(requests)
        val request = completionRequest()
        runtimeLoader.load(modelId, runDirectory, tokenizerPath).use { runtime ->
            val before = runtime.diagnostics()
            println("Model ID:             ${runtime.metadata.modelId}")
            println("Model kind:           ${runtime.metadata.kind.name.lowercase(Locale.ROOT)}")
            println("Checkpoint:           ${runtime.metadata.checkpointId}")
            println("Parameters:           ${runtime.metadata.parameterCount}")
            println("Concurrency:          ${runtime.metadata.concurrencyPolicy.name.lowercase(Locale.ROOT)}")
            println("Loaded once:          true")
            repeat(requests) { index ->
                val result = runtime.complete(request)
                println("Request ${index + 1}:            ${result.generatedText.quoted()}")
                println("Generated tokens:     ${result.generatedTokens}")
                println("KV cache:             ${result.metrics.cacheEnabled}")
                println("Prefill tokens:       ${result.metrics.prefillTokensProcessed}")
                println("Decode tokens:        ${result.metrics.decodeTokensProcessed}")
                println("Prompt tokens dropped:${result.metrics.promptTokensDiscarded.toString().padStart(2)}")
                println("Cache invalidations:  ${result.metrics.cacheInvalidations}")
                println("Model tokens/s:       ${result.metrics.generatedTokensPerSecond.rate()}")
            }
            val after = runtime.diagnostics()
            println("Completed requests:   ${after.completedRequests}")
            println("Managed arrays stable:${(before.managedArrayCount == after.managedArrayCount).yesNo(prefix = " ")}")
        }
        println("Runtime closed:       true")
    }

    private fun completionRequest() = CompletionRequest(
        prompt = prompt,
        maxNewTokens = maxNewTokens,
        sampling = samplingOptions(strategyName, temperature, topK, topP),
        seed = seed,
        addBos = addBos,
        cacheEnabled = cacheEnabled,
        contextPolicy = contextPolicy(contextPolicyName),
    )
}

@Command(
    name = "benchmark",
    mixinStandardHelpOptions = true,
    description = ["Compare loading for every request with reusing one inference runtime."],
)
class InferenceBenchmarkCommand(
    private val runtimeLoader: InferenceRuntimeLoader = InferenceRuntimeLoader(),
    private val tokenizerLoader: TokenizerArtifactLoader = TokenizerArtifactLoader(),
) : Callable<Int> {
    @Option(names = ["--model-id"], defaultValue = DEFAULT_MODEL_ID)
    lateinit var modelId: String

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--prompt"], required = true, paramLabel = "<text>")
    lateinit var prompt: String

    @Option(names = ["--requests"], defaultValue = "100")
    var requests: Int = 100

    @Option(names = ["--max-new-tokens"], defaultValue = "1")
    var maxNewTokens: Int = 1

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Long = 42

    override fun call(): Int = inferenceCli {
        requirePositiveRequests(requests)
        val request = TokenGenerationRequest(
            promptTokenIds = tokenizerLoader.load(tokenizerPath).tokenizer.encode(prompt),
            maxNewTokens = maxNewTokens,
            sampling = SamplingOptions(strategy = SamplingStrategy.GREEDY),
            seed = seed,
            cacheEnabled = false,
        )
        var reference: IntArray? = null
        var outputsIdentical = true

        val reloadStarted = System.nanoTime()
        repeat(requests) {
            runtimeLoader.load(modelId, runDirectory, tokenizerPath).use { runtime ->
                val generated = runtime.generate(request).generatedTokenIds
                val expected = reference
                if (expected == null) reference = generated.copyOf() else outputsIdentical = outputsIdentical && expected.contentEquals(generated)
            }
        }
        val reloadNanos = System.nanoTime() - reloadStarted

        var resourcesStable = false
        val reusedStarted = System.nanoTime()
        runtimeLoader.load(modelId, runDirectory, tokenizerPath).use { runtime ->
            val before = runtime.diagnostics().managedArrayCount
            repeat(requests) {
                val generated = runtime.generate(request).generatedTokenIds
                outputsIdentical = outputsIdentical && requireNotNull(reference).contentEquals(generated)
            }
            resourcesStable = before == runtime.diagnostics().managedArrayCount
        }
        val reusedNanos = System.nanoTime() - reusedStarted

        println("Requests:                  $requests")
        println("Legacy-style model loads:  $requests")
        println("Reused-runtime model loads: 1")
        println("Reload-each elapsed:        ${reloadNanos.milliseconds()} ms")
        println("Reuse-one elapsed:          ${reusedNanos.milliseconds()} ms")
        println("Observed speedup:           ${"%.2f".format(Locale.ROOT, reloadNanos.toDouble() / max(1L, reusedNanos))}x")
        println("Outputs identical:         $outputsIdentical")
        println("Managed arrays stable:     $resourcesStable")
        println("Runtime closed:            true")
    }
}

@Command(
    name = "cache-benchmark",
    mixinStandardHelpOptions = true,
    description = ["Compare cached decoding with full recomputation for several generation lengths."],
)
class InferenceCacheBenchmarkCommand(
    private val runtimeLoader: InferenceRuntimeLoader = InferenceRuntimeLoader(),
    private val tokenizerLoader: TokenizerArtifactLoader = TokenizerArtifactLoader(),
) : Callable<Int> {
    @Option(names = ["--model-id"], defaultValue = DEFAULT_MODEL_ID)
    lateinit var modelId: String

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--prompt"], required = true, paramLabel = "<text>")
    lateinit var prompt: String

    @Option(names = ["--new-token-counts"], split = ",", defaultValue = "32,64,128")
    lateinit var newTokenCounts: IntArray

    @Option(names = ["--iterations"], defaultValue = "3")
    var iterations: Int = 3

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Long = 42

    override fun call(): Int = inferenceCli {
        requirePositiveRequests(iterations)
        if (newTokenCounts.isEmpty() || newTokenCounts.any { count -> count <= 0 }) {
            throw InferenceException("--new-token-counts must contain positive integers")
        }
        val promptTokenIds = tokenizerLoader.load(tokenizerPath).tokenizer.encode(prompt)
        runtimeLoader.load(modelId, runDirectory, tokenizerPath).use { runtime ->
            println("Model ID:              ${runtime.metadata.modelId}")
            println("Context length:        ${runtime.metadata.contextLength}")
            println("Prompt tokens:         ${promptTokenIds.size}")
            println("Iterations:            $iterations")
            println("newTokens cacheTok/s fullTok/s speedup cacheModelTokens fullModelTokens outputsIdentical")
            newTokenCounts.forEach { newTokenCount ->
                val cached = measure(runtime, promptTokenIds, newTokenCount, cacheEnabled = true)
                val full = measure(runtime, promptTokenIds, newTokenCount, cacheEnabled = false)
                val outputsIdentical = cached.generatedTokenIds.contentEquals(full.generatedTokenIds)
                println(
                    "%9d %10s %9s %7sx %16d %15d %s".format(
                        Locale.ROOT,
                        newTokenCount,
                        cached.tokensPerSecond.rate(),
                        full.tokensPerSecond.rate(),
                        cached.tokensPerSecond.div(maxOf(full.tokensPerSecond, Double.MIN_VALUE)).ratio(),
                        cached.modelTokensProcessed,
                        full.modelTokensProcessed,
                        outputsIdentical,
                    ),
                )
            }
        }
        println("Runtime closed:        true")
    }

    private fun measure(
        runtime: io.github.lxptechnologies.lxpmini.inference.InferenceRuntime,
        promptTokenIds: IntArray,
        newTokenCount: Int,
        cacheEnabled: Boolean,
    ): CacheBenchmarkMeasurement {
        val request = TokenGenerationRequest(
            promptTokenIds = promptTokenIds,
            maxNewTokens = newTokenCount,
            sampling = SamplingOptions(strategy = SamplingStrategy.GREEDY),
            seed = seed,
            cacheEnabled = cacheEnabled,
            contextPolicy = ContextOverflowPolicy.REJECT,
        )
        runtime.generateWithMetrics(request)
        var totalNanos = 0L
        var modelTokensProcessed = 0L
        var generatedTokenIds = IntArray(0)
        var generatedTokenCount = 0
        repeat(iterations) {
            val result = runtime.generateWithMetrics(request)
            totalNanos += result.metrics.totalModelNanos
            modelTokensProcessed = result.metrics.modelTokensProcessed
            generatedTokenIds = result.generation.generatedTokenIds
            generatedTokenCount = result.generation.generatedTokenIds.size
        }
        val averageNanos = totalNanos.toDouble() / iterations
        val tokensPerSecond = if (averageNanos == 0.0) 0.0 else generatedTokenCount * 1_000_000_000.0 / averageNanos
        return CacheBenchmarkMeasurement(tokensPerSecond, modelTokensProcessed, generatedTokenIds)
    }
}

private data class CacheBenchmarkMeasurement(
    val tokensPerSecond: Double,
    val modelTokensProcessed: Long,
    val generatedTokenIds: IntArray,
)

private const val DEFAULT_MODEL_ID = "lxp-mini-1xm-base"

private fun inferenceCli(action: () -> Unit): Int = try {
    action()
    0
} catch (exception: InferenceException) {
    System.err.println("Inference error: ${exception.message}")
    2
} catch (exception: GenerationException) {
    System.err.println("Generation error: ${exception.message}")
    2
} catch (exception: TokenizerException) {
    System.err.println("Tokenizer error: ${exception.message}")
    2
}

private fun requirePositiveRequests(requests: Int) {
    if (requests <= 0) throw InferenceException("--requests must be positive")
}

private fun samplingOptions(
    strategyName: String,
    temperature: Double,
    topK: Int,
    topP: Double,
): SamplingOptions {
    val strategy = when (strategyName.lowercase(Locale.ROOT)) {
        "greedy" -> SamplingStrategy.GREEDY
        "sample" -> SamplingStrategy.SAMPLE
        else -> throw InferenceException("--strategy must be 'greedy' or 'sample'")
    }
    return SamplingOptions(strategy, temperature, topK, topP)
}

private fun contextPolicy(name: String): ContextOverflowPolicy = when (name.lowercase(Locale.ROOT)) {
    "reject" -> ContextOverflowPolicy.REJECT
    "sliding-window" -> ContextOverflowPolicy.SLIDING_WINDOW
    else -> throw InferenceException("--context-policy must be 'sliding-window' or 'reject'")
}

private fun String.quoted(): String = buildString {
    append('"')
    for (character in this@quoted) {
        when (character) {
            '\n' -> append("\\n")
            '\r' -> append("\\r")
            '\t' -> append("\\t")
            '"' -> append("\\\"")
            '\\' -> append("\\\\")
            else -> if (character.isISOControl()) append("\\u%04x".format(character.code)) else append(character)
        }
    }
    append('"')
}

private fun Boolean.yesNo(prefix: String = ""): String = prefix + if (this) "true" else "false"
private fun Long.milliseconds(): String = "%.2f".format(Locale.ROOT, this / 1_000_000.0)
private fun Double.rate(): String = "%.2f".format(Locale.ROOT, this)
private fun Double.ratio(): String = "%.2f".format(Locale.ROOT, this)
