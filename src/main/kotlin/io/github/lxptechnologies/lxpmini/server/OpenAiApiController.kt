package io.github.lxptechnologies.lxpmini.server

import com.fasterxml.jackson.databind.ObjectMapper
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.inference.CompletionRequest
import io.github.lxptechnologies.lxpmini.inference.CompletionResult
import io.github.lxptechnologies.lxpmini.inference.ContextOverflowPolicy
import org.springframework.http.CacheControl
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.nio.charset.StandardCharsets
import java.time.Instant
import java.util.UUID

@RestController
class OpenAiApiController(
    private val inference: LocalInferenceService,
    private val capabilities: ServerCapabilities,
    private val objectMapper: ObjectMapper,
) {
    @GetMapping("/health")
    fun health() = HealthResponse(
        status = "ok",
        model = inference.metadata.modelId,
        checkpoint = inference.metadata.checkpointId,
        streamingEnabled = capabilities.streamingEnabled,
    )

    @GetMapping("/v1/models")
    fun models() = ModelListResponse(data = listOf(modelResponse()))

    @GetMapping("/v1/models/{model}")
    fun model(@PathVariable model: String): ModelResponse {
        requireModel(model)
        return modelResponse()
    }

    @PostMapping("/v1/completions", consumes = [MediaType.APPLICATION_JSON_VALUE])
    fun complete(@RequestBody request: CompletionCreateRequest): ResponseEntity<StreamingResponseBody> {
        val validated = validate(request)
        val completionId = "cmpl-${UUID.randomUUID().toString().replace("-", "")}" 
        val created = Instant.now().epochSecond
        return if (request.stream) {
            streamingResponse(validated, completionId, created)
        } else {
            val result = inference.complete(validated.runtimeRequest)
            val response = completionResponse(completionId, created, result)
            val body = StreamingResponseBody { output -> objectMapper.writeValue(output, response) }
            ResponseEntity.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
        }
    }

    private fun streamingResponse(
        validated: ValidatedCompletionRequest,
        completionId: String,
        created: Long,
    ): ResponseEntity<StreamingResponseBody> {
        val body = StreamingResponseBody { output ->
            val writer = BufferedWriter(OutputStreamWriter(output, StandardCharsets.UTF_8))
            val result = inference.completeStreaming(validated.runtimeRequest) { delta ->
                writeSse(writer, completionChunk(completionId, created, delta, finishReason = null))
            }
            writeSse(writer, completionChunk(completionId, created, "", finishReason(result)))
            writer.write("data: [DONE]\n\n")
            writer.flush()
        }
        return ResponseEntity.ok()
            .contentType(MediaType.TEXT_EVENT_STREAM)
            .cacheControl(CacheControl.noCache())
            .header("X-Content-Type-Options", "nosniff")
            .body(body)
    }

    private fun validate(request: CompletionCreateRequest): ValidatedCompletionRequest {
        val model = request.model?.takeIf(String::isNotBlank)
            ?: invalid("model is required", "model", "missing_required_parameter")
        requireModel(model)
        val prompt = request.prompt ?: invalid("prompt is required", "prompt", "missing_required_parameter")
        if (prompt.isEmpty()) invalid("prompt must not be empty", "prompt", "invalid_prompt")
        if (request.maxTokens < 0) invalid("max_tokens must be non-negative", "max_tokens", "invalid_value")
        if (!request.temperature.isFinite() || request.temperature !in 0.0..2.0) {
            invalid("temperature must be finite and in [0, 2]", "temperature", "invalid_value")
        }
        if (!request.topP.isFinite() || request.topP <= 0.0 || request.topP > 1.0) {
            invalid("top_p must be finite and in (0, 1]", "top_p", "invalid_value")
        }
        if (request.n != 1) unsupported("Only n=1 is supported", "n")
        if (request.echo) unsupported("echo=true is not supported", "echo")
        if (request.bestOf != null && request.bestOf != 1) unsupported("Only best_of=1 is supported", "best_of")
        if (request.logprobs != null) unsupported("logprobs is not supported", "logprobs")
        if (request.stop != null) unsupported("stop sequences are not supported yet", "stop")
        if (request.frequencyPenalty != 0.0) unsupported("frequency_penalty is not supported", "frequency_penalty")
        if (request.presencePenalty != 0.0) unsupported("presence_penalty is not supported", "presence_penalty")
        if (request.logitBias != null) unsupported("logit_bias is not supported", "logit_bias")
        if (request.streamOptions != null) unsupported("stream_options is not supported", "stream_options")
        if (request.suffix != null) unsupported("suffix is not supported", "suffix")
        if (request.user != null) unsupported("user is not supported", "user")
        if (request.stream && !capabilities.streamingEnabled) {
            unsupported("Streaming is disabled on this server", "stream")
        }
        val promptTokens = inference.countPromptTokens(prompt)
        if (promptTokens.toLong() + request.maxTokens > inference.metadata.contextLength) {
            invalid(
                "This request needs ${promptTokens + request.maxTokens} tokens, but the model context is " +
                    "${inference.metadata.contextLength}",
                "max_tokens",
                "context_length_exceeded",
            )
        }
        val strategy = if (request.temperature == 0.0) SamplingStrategy.GREEDY else SamplingStrategy.SAMPLE
        val effectiveTemperature = if (strategy == SamplingStrategy.GREEDY) 1.0 else request.temperature
        return ValidatedCompletionRequest(
            runtimeRequest = CompletionRequest(
                prompt = prompt,
                maxNewTokens = request.maxTokens,
                sampling = SamplingOptions(strategy, effectiveTemperature, topP = request.topP),
                seed = request.seed,
                cacheEnabled = true,
                contextPolicy = ContextOverflowPolicy.REJECT,
            ),
        )
    }

    private fun completionResponse(id: String, created: Long, result: CompletionResult) = CompletionResponse(
        id = id,
        created = created,
        model = inference.metadata.modelId,
        choices = listOf(
            CompletionChoiceResponse(
                text = result.generatedText,
                finishReason = finishReason(result),
            ),
        ),
        usage = CompletionUsageResponse(result.promptTokens, result.generatedTokens, result.totalTokens),
        systemFingerprint = systemFingerprint(),
    )

    private fun completionChunk(
        id: String,
        created: Long,
        text: String,
        finishReason: String?,
    ) = CompletionChunkResponse(
        id = id,
        created = created,
        model = inference.metadata.modelId,
        choices = listOf(CompletionChoiceResponse(text = text, finishReason = finishReason)),
        systemFingerprint = systemFingerprint(),
    )

    private fun modelResponse() = ModelResponse(
        id = inference.metadata.modelId,
        created = inference.metadata.createdAtEpochSeconds,
    )

    private fun requireModel(model: String) {
        if (model != inference.metadata.modelId) {
            throw OpenAiApiException(
                status = HttpStatus.NOT_FOUND,
                message = "The model '$model' does not exist on this server",
                param = "model",
                code = "model_not_found",
            )
        }
    }

    private fun finishReason(result: CompletionResult): String = if (result.stoppedByEos) "stop" else "length"

    private fun systemFingerprint(): String = "lxp-${inference.metadata.checkpointSha256.take(12)}"

    private fun writeSse(writer: BufferedWriter, chunk: CompletionChunkResponse) {
        writer.write("data: ")
        writer.write(objectMapper.writeValueAsString(chunk))
        writer.write("\n\n")
        writer.flush()
    }

    private fun invalid(message: String, param: String, code: String): Nothing =
        throw OpenAiApiException(HttpStatus.BAD_REQUEST, message, param = param, code = code)

    private fun unsupported(message: String, param: String): Nothing =
        throw OpenAiApiException(HttpStatus.BAD_REQUEST, message, param = param, code = "unsupported_feature")
}

private data class ValidatedCompletionRequest(val runtimeRequest: CompletionRequest)
