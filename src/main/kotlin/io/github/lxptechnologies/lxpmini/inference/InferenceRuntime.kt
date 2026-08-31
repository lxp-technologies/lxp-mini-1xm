package io.github.lxptechnologies.lxpmini.inference

import ai.djl.ndarray.NDManager
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.generation.AutoregressiveGenerator
import io.github.lxptechnologies.lxpmini.generation.GenerationResult
import io.github.lxptechnologies.lxpmini.generation.GenerationStep
import io.github.lxptechnologies.lxpmini.generation.GenerationTokenConstraint
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.TokenSampler
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.runtime.RuntimeDeviceResolver
import io.github.lxptechnologies.lxpmini.runtime.RuntimeDeviceSelector
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import io.github.lxptechnologies.lxpmini.tokenizer.Tokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerArtifactLoader
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import java.nio.file.Path
import java.time.Instant
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class InferenceRuntimeLoader(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
    private val runStore: RunStore = RunStore(),
    private val tokenizerLoader: TokenizerArtifactLoader = TokenizerArtifactLoader(),
    private val deviceResolver: RuntimeDeviceSelector = RuntimeDeviceResolver(),
) {
    fun load(
        modelId: String,
        runDirectory: Path,
        tokenizerPath: Path,
        deviceOverride: String? = null,
    ): InferenceRuntime {
        requireModelId(modelId)
        var rootManager: NDManager? = null
        var modelManager: NDManager? = null
        var model: DecoderLanguageModel? = null
        try {
            val configPath = runDirectory.resolve(RunStore.CONFIG_FILE)
            val config = configLoader.load(configPath)
            val device = deviceResolver.resolve(deviceOverride ?: config.runtime.device)
            val root = NDManager.newBaseManager(device.selected)
            rootManager = root
            val modelScope = root.newSubManager()
            modelManager = modelScope
            root.name = "inference-runtime-$modelId"
            modelScope.name = "inference-model-$modelId"
            val runMetadata = runStore.loadMetadata(runDirectory)
            val configSha256 = Sha256.of(configPath)
            if (runMetadata.configSha256 != configSha256) {
                throw InferenceException(
                    "Run configuration checksum mismatch: expected ${runMetadata.configSha256}, got $configSha256",
                )
            }
            val tokenizerArtifact = tokenizerLoader.load(tokenizerPath)
            if (tokenizerArtifact.tokenizer.vocabularySize != config.model.vocabSize) {
                throw InferenceException(
                    "Tokenizer vocabulary ${tokenizerArtifact.tokenizer.vocabularySize} " +
                        "does not match model vocabulary ${config.model.vocabSize}",
                )
            }
            if (runMetadata.tokenizer != RunStore.DEFAULT_TOKENIZER && runMetadata.tokenizer != tokenizerArtifact.type) {
                throw InferenceException(
                    "Run expects tokenizer type '${runMetadata.tokenizer}', got '${tokenizerArtifact.type}'",
                )
            }
            runMetadata.tokenizerSha256?.let { expected ->
                val actual = Sha256.of(tokenizerPath)
                if (actual != expected) {
                    throw InferenceException("Tokenizer checksum mismatch: expected $expected, got $actual")
                }
            }

            model = DecoderLanguageModel(modelScope, config.model)
            val checkpoint = checkpointStore.loadLatest(runDirectory, model, modelScope, configSha256)
            modelScope.cap()
            return InferenceRuntime(
                metadata = InferenceModelMetadata(
                    modelId = modelId,
                    kind = InferenceModelKind.BASE,
                    checkpointId = checkpoint.manifest.checkpointId,
                    checkpointSha256 = checkpoint.manifest.modelSha256,
                    vocabularySize = config.model.vocabSize,
                    contextLength = config.model.contextLength,
                    parameterCount = model.actualParameterCount(),
                    tokenizerType = tokenizerArtifact.type,
                    concurrencyPolicy = InferenceConcurrencyPolicy.SERIALIZED,
                    createdAtEpochSeconds = Instant.parse(runMetadata.createdAtUtc).epochSecond,
                    requestedDevice = device.requested.value,
                    selectedDevice = device.selectedName,
                    engineName = device.engineName,
                    nativeRuntimeVersion = device.nativeRuntimeVersion,
                    gpuCount = device.gpuCount,
                ),
                tokenizer = tokenizerArtifact.tokenizer,
                rootManager = root,
                modelManager = modelScope,
                model = model,
            )
        } catch (throwable: Throwable) {
            closeAfterFailedLoad(throwable, model, modelManager, rootManager)
            if (throwable is Error) throw throwable
            if (throwable is InferenceException) throw throwable
            throw InferenceException("Cannot load inference runtime '$modelId': ${throwable.message}", throwable)
        }
    }

    private fun requireModelId(modelId: String) {
        if (!modelId.matches(MODEL_ID_PATTERN)) {
            throw InferenceException(
                "modelId must match ${MODEL_ID_PATTERN.pattern} and contain at most 64 characters",
            )
        }
    }

    private fun closeAfterFailedLoad(
        failure: Throwable,
        model: DecoderLanguageModel?,
        modelManager: NDManager?,
        rootManager: NDManager?,
    ) {
        listOfNotNull<AutoCloseable>(model, modelManager, rootManager).forEach { resource ->
            try {
                resource.close()
            } catch (closeFailure: Throwable) {
                failure.addSuppressed(closeFailure)
            }
        }
    }

    private companion object {
        val MODEL_ID_PATTERN = Regex("[a-z0-9][a-z0-9._-]{0,63}")
    }
}

class InferenceRuntime internal constructor(
    val metadata: InferenceModelMetadata,
    private val tokenizer: Tokenizer,
    private val rootManager: NDManager,
    private val modelManager: NDManager,
    private val model: DecoderLanguageModel,
) : AutoCloseable {
    private val parameterStore = ParameterStore(modelManager, false)
    private val lifecycleLock = ReentrantLock(true)
    private val closed = AtomicBoolean(false)
    private val completedRequests = AtomicLong(0)
    private val utf8TokenConstraint = Utf8TokenConstraint(
        tokenizer = tokenizer,
        vocabularySize = metadata.vocabularySize,
        eosTokenId = SpecialToken.EOS.id,
    )

    val isClosed: Boolean
        get() = closed.get()

    fun countPromptTokens(prompt: String, addBos: Boolean = false): Int = lifecycleLock.withLock {
        requireOpen()
        tokenizer.encode(prompt, addBos = addBos).size
    }

    fun generate(request: TokenGenerationRequest): GenerationResult = generateWithMetrics(request).generation

    fun generateWithMetrics(request: TokenGenerationRequest): InferenceGenerationResult = generateInternal(request)

    fun generateStreaming(
        request: TokenGenerationRequest,
        onStep: (GenerationStep) -> Unit,
    ): InferenceGenerationResult = generateInternal(request, onStep)

    private fun generateInternal(
        request: TokenGenerationRequest,
        onStep: (GenerationStep) -> Unit = {},
        tokenConstraint: GenerationTokenConstraint? = null,
    ): InferenceGenerationResult = lifecycleLock.withLock {
        requireOpen()
        validateContextBudget(request)
        val requestNumber = completedRequests.get() + 1
        rootManager.newSubManager().use { requestManager ->
            requestManager.name = "inference-request-$requestNumber"
            newLogitsSession(request, requestManager).use { session ->
                val generator = AutoregressiveGenerator(
                    metadata.contextLength,
                    metadata.vocabularySize,
                    TokenSampler(request.seed),
                    tokenConstraint = tokenConstraint,
                    logitsProvider = session::lastTokenLogits,
                )
                val result = generator.generate(
                    request.promptTokenIds,
                    request.maxNewTokens,
                    request.eosTokenId,
                    request.sampling,
                    onStep,
                )
                completedRequests.incrementAndGet()
                InferenceGenerationResult(
                    result,
                    session.metrics(
                        result.generatedTokenIds.size,
                        (request.promptTokenIds.size - metadata.contextLength).coerceAtLeast(0),
                    ),
                )
            }
        }
    }

    fun complete(request: CompletionRequest): CompletionResult {
        val promptTokenIds = encodePrompt(request)
        val generationRequest = request.toTokenGenerationRequest(promptTokenIds)
        val detailedGeneration = generateInternal(
            generationRequest,
            tokenConstraint = utf8TokenConstraint,
        )
        return completionResult(request, detailedGeneration.generation, detailedGeneration.metrics)
    }

    fun completeStreaming(
        request: CompletionRequest,
        onTextDelta: (String) -> Unit,
    ): CompletionResult = completeInternal(request, onTextDelta)

    private fun completeInternal(
        request: CompletionRequest,
        onTextDelta: (String) -> Unit,
    ): CompletionResult {
        val promptTokenIds = encodePrompt(request)
        val textDecoder = IncrementalTextDecoder(tokenizer)
        val generationRequest = request.toTokenGenerationRequest(promptTokenIds)
        val detailedGeneration = generateInternal(
            request = generationRequest,
            onStep = { step -> textDecoder.accept(step.sampling.tokenId, onTextDelta) },
            tokenConstraint = utf8TokenConstraint,
        )
        val generation = detailedGeneration.generation
        textDecoder.finish(onTextDelta)
        return completionResult(request, generation, detailedGeneration.metrics)
    }

    private fun encodePrompt(request: CompletionRequest): IntArray {
        val promptTokenIds = tokenizer.encode(request.prompt, addBos = request.addBos)
        if (promptTokenIds.isEmpty()) {
            throw InferenceException("Prompt must produce at least one token; provide text or enable addBos")
        }
        return promptTokenIds
    }

    private fun CompletionRequest.toTokenGenerationRequest(promptTokenIds: IntArray) = TokenGenerationRequest(
        promptTokenIds = promptTokenIds,
        maxNewTokens = maxNewTokens,
        sampling = sampling,
        seed = seed,
        cacheEnabled = cacheEnabled,
        contextPolicy = contextPolicy,
    )

    private fun completionResult(
        request: CompletionRequest,
        generation: GenerationResult,
        metrics: InferenceMetrics,
    ): CompletionResult = try {
        CompletionResult(
            modelId = metadata.modelId,
            checkpointId = metadata.checkpointId,
            prompt = request.prompt,
            generatedText = tokenizer.decodeLossy(generation.generatedTokenIds),
            completeText = tokenizer.decodeLossy(generation.allTokenIds),
            promptTokens = generation.promptTokenIds.size,
            generatedTokens = generation.generatedTokenIds.size,
            stoppedByEos = generation.stoppedByEos,
            generation = generation,
            metrics = metrics,
        )
    } catch (exception: TokenizerException) {
        throw InferenceException(
            "Generated token IDs cannot be decoded: ${generation.generatedTokenIds.contentToString()}",
            exception,
        )
    }

    fun diagnostics(): InferenceRuntimeDiagnostics = lifecycleLock.withLock {
        requireOpen()
        InferenceRuntimeDiagnostics(
            completedRequests = completedRequests.get(),
            managedArrayCount = modelManager.managedArrays.size,
            concurrencyPolicy = metadata.concurrencyPolicy,
        )
    }

    override fun close() {
        lifecycleLock.withLock {
            if (!closed.compareAndSet(false, true)) return
            var failure: Throwable? = null
            try {
                model.close()
            } catch (throwable: Throwable) {
                failure = throwable
            }
            try {
                modelManager.close()
            } catch (throwable: Throwable) {
                if (failure == null) failure = throwable else failure.addSuppressed(throwable)
            }
            try {
                rootManager.close()
            } catch (throwable: Throwable) {
                if (failure == null) failure = throwable else failure.addSuppressed(throwable)
            }
            failure?.let { throw it }
        }
    }

    private fun newLogitsSession(request: TokenGenerationRequest, requestManager: NDManager): InferenceLogitsSession =
        if (request.cacheEnabled) {
            KeyValueLogitsSession(requestManager, model, parameterStore, request.contextPolicy)
        } else {
            FullRecomputeLogitsSession(requestManager, model, parameterStore, request.contextPolicy)
        }

    private fun validateContextBudget(request: TokenGenerationRequest) {
        if (request.contextPolicy == ContextOverflowPolicy.REJECT &&
            request.promptTokenIds.size.toLong() + request.maxNewTokens > metadata.contextLength
        ) {
            throw InferenceException(
                "Prompt (${request.promptTokenIds.size}) plus maxNewTokens (${request.maxNewTokens}) " +
                    "exceeds context ${metadata.contextLength} with REJECT policy",
            )
        }
    }

    private fun requireOpen() {
        if (closed.get()) throw InferenceException("Inference runtime '${metadata.modelId}' is closed")
    }
}

enum class InferenceModelKind {
    BASE,
}

enum class InferenceConcurrencyPolicy {
    SERIALIZED,
}

data class InferenceModelMetadata(
    val modelId: String,
    val kind: InferenceModelKind,
    val checkpointId: String,
    val checkpointSha256: String,
    val vocabularySize: Int,
    val contextLength: Int,
    val parameterCount: Long,
    val tokenizerType: String,
    val concurrencyPolicy: InferenceConcurrencyPolicy,
    val createdAtEpochSeconds: Long,
    val requestedDevice: String,
    val selectedDevice: String,
    val engineName: String,
    val nativeRuntimeVersion: String,
    val gpuCount: Int,
)

data class TokenGenerationRequest(
    val promptTokenIds: IntArray,
    val maxNewTokens: Int = 16,
    val sampling: SamplingOptions = SamplingOptions(),
    val seed: Long = 42,
    val eosTokenId: Int = SpecialToken.EOS.id,
    val cacheEnabled: Boolean = true,
    val contextPolicy: ContextOverflowPolicy = ContextOverflowPolicy.SLIDING_WINDOW,
)

data class CompletionRequest(
    val prompt: String,
    val maxNewTokens: Int = 16,
    val sampling: SamplingOptions = SamplingOptions(),
    val seed: Long = 42,
    val addBos: Boolean = false,
    val cacheEnabled: Boolean = true,
    val contextPolicy: ContextOverflowPolicy = ContextOverflowPolicy.SLIDING_WINDOW,
)

data class CompletionResult(
    val modelId: String,
    val checkpointId: String,
    val prompt: String,
    val generatedText: String,
    val completeText: String,
    val promptTokens: Int,
    val generatedTokens: Int,
    val stoppedByEos: Boolean,
    val generation: GenerationResult,
    val metrics: InferenceMetrics,
) {
    val totalTokens: Int = promptTokens + generatedTokens
}

data class InferenceGenerationResult(
    val generation: GenerationResult,
    val metrics: InferenceMetrics,
)

enum class ContextOverflowPolicy {
    REJECT,
    SLIDING_WINDOW,
}

data class InferenceMetrics(
    val cacheEnabled: Boolean,
    val contextPolicy: ContextOverflowPolicy,
    val prefillTokensProcessed: Long,
    val decodeTokensProcessed: Long,
    val prefillNanos: Long,
    val decodeNanos: Long,
    val cacheInvalidations: Int,
    val peakCachedTokens: Int,
    val generatedTokenCount: Int,
    val promptTokensDiscarded: Int,
) {
    val totalModelNanos: Long = prefillNanos + decodeNanos
    val modelTokensProcessed: Long = prefillTokensProcessed + decodeTokensProcessed
    val generatedTokensPerSecond: Double =
        if (totalModelNanos == 0L) 0.0 else generatedTokenCount * 1_000_000_000.0 / totalModelNanos
}

data class InferenceRuntimeDiagnostics(
    val completedRequests: Long,
    val managedArrayCount: Int,
    val concurrencyPolicy: InferenceConcurrencyPolicy,
)
