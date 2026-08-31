package io.github.lxptechnologies.lxpmini.server

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.JsonNode

data class HealthResponse(
    val status: String,
    val model: String,
    val checkpoint: String,
    @JsonProperty("streaming_enabled") val streamingEnabled: Boolean,
)

data class ModelListResponse(
    val `object`: String = "list",
    val data: List<ModelResponse>,
)

data class ModelResponse(
    val id: String,
    val `object`: String = "model",
    val created: Long,
    @JsonProperty("owned_by") val ownedBy: String = "lxp-technologies",
)

data class CompletionCreateRequest(
    val model: String? = null,
    val prompt: String? = null,
    @JsonProperty("max_tokens") val maxTokens: Int = 16,
    val temperature: Double = 1.0,
    @JsonProperty("top_p") val topP: Double = 1.0,
    val seed: Long = 42,
    val stream: Boolean = false,
    val stop: JsonNode? = null,
    val n: Int = 1,
    val echo: Boolean = false,
    val logprobs: Int? = null,
    @JsonProperty("best_of") val bestOf: Int? = null,
    @JsonProperty("frequency_penalty") val frequencyPenalty: Double = 0.0,
    @JsonProperty("presence_penalty") val presencePenalty: Double = 0.0,
    @JsonProperty("logit_bias") val logitBias: JsonNode? = null,
    @JsonProperty("stream_options") val streamOptions: JsonNode? = null,
    val suffix: String? = null,
    val user: String? = null,
)

data class CompletionResponse(
    val id: String,
    val `object`: String = "text_completion",
    val created: Long,
    val model: String,
    val choices: List<CompletionChoiceResponse>,
    val usage: CompletionUsageResponse,
    @JsonProperty("system_fingerprint") val systemFingerprint: String,
)

data class CompletionChunkResponse(
    val id: String,
    val `object`: String = "text_completion",
    val created: Long,
    val model: String,
    val choices: List<CompletionChoiceResponse>,
    @JsonProperty("system_fingerprint") val systemFingerprint: String,
)

data class CompletionChoiceResponse(
    val text: String,
    val index: Int = 0,
    val logprobs: Nothing? = null,
    @JsonProperty("finish_reason") val finishReason: String?,
)

data class CompletionUsageResponse(
    @JsonProperty("prompt_tokens") val promptTokens: Int,
    @JsonProperty("completion_tokens") val completionTokens: Int,
    @JsonProperty("total_tokens") val totalTokens: Int,
)

data class OpenAiErrorResponse(val error: OpenAiError)

data class OpenAiError(
    val message: String,
    val type: String,
    val param: String?,
    val code: String,
)
