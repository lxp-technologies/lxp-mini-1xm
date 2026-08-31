package io.github.lxptechnologies.lxpmini.server

import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.inference.CompletionRequest
import io.github.lxptechnologies.lxpmini.inference.ContextOverflowPolicy
import org.springframework.core.io.ClassPathResource
import org.springframework.core.io.Resource
import org.springframework.http.HttpStatus
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestBody
import org.springframework.web.bind.annotation.RestController

@RestController
class PlaygroundController(
    private val inference: LocalInferenceService,
    private val formatter: ChatPromptFormatter,
) {
    @GetMapping("/", produces = [MediaType.TEXT_HTML_VALUE])
    fun index(): ResponseEntity<Resource> = ResponseEntity.ok()
        .contentType(MediaType.TEXT_HTML)
        .body(ClassPathResource("web/index.html"))

    @PostMapping(
        "/playground/completions",
        consumes = [MediaType.APPLICATION_JSON_VALUE],
        produces = [MediaType.APPLICATION_JSON_VALUE],
    )
    fun complete(@RequestBody request: PlaygroundCompletionRequest): PlaygroundCompletionResponse {
        if (request.userMessage.isBlank()) invalid("user_message must not be blank", "user_message")
        if (request.maxNewTokens <= 0) invalid("max_new_tokens must be positive", "max_new_tokens")
        if (!request.temperature.isFinite() || request.temperature !in 0.0..2.0) {
            invalid("temperature must be finite and in [0, 2]", "temperature")
        }
        if (!request.topP.isFinite() || request.topP <= 0.0 || request.topP > 1.0) {
            invalid("top_p must be finite and in (0, 1]", "top_p")
        }
        if (request.topK < 0) invalid("top_k must be non-negative", "top_k")
        request.messages.forEach { message ->
            if (message.content.isBlank()) invalid("message content must not be blank", "messages.content")
        }
        val prompt = formatter.format(request.systemPrompt, request.messages, request.userMessage)
        val promptTokens = inference.countPromptTokens(prompt)
        if (promptTokens.toLong() + request.maxNewTokens > inference.metadata.contextLength) {
            throw OpenAiApiException(
                HttpStatus.BAD_REQUEST,
                "The formatted conversation needs ${promptTokens + request.maxNewTokens} tokens, but the model " +
                    "context is ${inference.metadata.contextLength}",
                param = "max_new_tokens",
                code = "context_length_exceeded",
            )
        }
        val strategy = if (request.temperature == 0.0) SamplingStrategy.GREEDY else SamplingStrategy.SAMPLE
        val temperature = if (strategy == SamplingStrategy.GREEDY) 1.0 else request.temperature
        val result = inference.complete(
            CompletionRequest(
                prompt = prompt,
                maxNewTokens = request.maxNewTokens,
                sampling = SamplingOptions(strategy, temperature, request.topK, request.topP),
                seed = request.seed,
                cacheEnabled = true,
                contextPolicy = ContextOverflowPolicy.REJECT,
            ),
        )
        return PlaygroundCompletionResponse(
            text = result.generatedText,
            formattedPrompt = prompt,
            promptTokens = result.promptTokens,
            completionTokens = result.generatedTokens,
        )
    }

    private fun invalid(message: String, param: String): Nothing = throw OpenAiApiException(
        HttpStatus.BAD_REQUEST,
        message,
        param = param,
        code = "invalid_value",
    )
}

data class PlaygroundCompletionRequest(
    @com.fasterxml.jackson.annotation.JsonProperty("system_prompt") val systemPrompt: String? = null,
    val messages: List<PlaygroundMessage> = emptyList(),
    @com.fasterxml.jackson.annotation.JsonProperty("user_message") val userMessage: String,
    val temperature: Double = 0.0,
    @com.fasterxml.jackson.annotation.JsonProperty("max_new_tokens") val maxNewTokens: Int = 32,
    @com.fasterxml.jackson.annotation.JsonProperty("top_p") val topP: Double = 0.95,
    @com.fasterxml.jackson.annotation.JsonProperty("top_k") val topK: Int = 0,
    val seed: Long = 42,
)

data class PlaygroundMessage(val role: String, val content: String)

data class PlaygroundCompletionResponse(
    val text: String,
    @com.fasterxml.jackson.annotation.JsonProperty("formatted_prompt") val formattedPrompt: String,
    @com.fasterxml.jackson.annotation.JsonProperty("prompt_tokens") val promptTokens: Int,
    @com.fasterxml.jackson.annotation.JsonProperty("completion_tokens") val completionTokens: Int,
)
