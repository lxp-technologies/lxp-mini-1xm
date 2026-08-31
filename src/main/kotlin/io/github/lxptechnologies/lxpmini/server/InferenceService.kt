package io.github.lxptechnologies.lxpmini.server

import io.github.lxptechnologies.lxpmini.inference.CompletionRequest
import io.github.lxptechnologies.lxpmini.inference.CompletionResult
import io.github.lxptechnologies.lxpmini.inference.InferenceModelMetadata
import io.github.lxptechnologies.lxpmini.inference.InferenceRuntime

interface LocalInferenceService : AutoCloseable {
    val metadata: InferenceModelMetadata
    fun countPromptTokens(prompt: String): Int
    fun complete(request: CompletionRequest): CompletionResult
    fun completeStreaming(request: CompletionRequest, onTextDelta: (String) -> Unit): CompletionResult
}

class RuntimeInferenceService(
    private val runtime: InferenceRuntime,
) : LocalInferenceService {
    override val metadata: InferenceModelMetadata
        get() = runtime.metadata

    override fun countPromptTokens(prompt: String): Int = runtime.countPromptTokens(prompt)

    override fun complete(request: CompletionRequest): CompletionResult = runtime.complete(request)

    override fun completeStreaming(
        request: CompletionRequest,
        onTextDelta: (String) -> Unit,
    ): CompletionResult = runtime.completeStreaming(request, onTextDelta)

    override fun close() = runtime.close()
}

data class ServerCapabilities(val streamingEnabled: Boolean)
