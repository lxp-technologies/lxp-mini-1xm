package io.github.lxptechnologies.lxpmini.server

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.module.kotlin.jacksonObjectMapper
import io.github.lxptechnologies.lxpmini.cli.CheckpointDemoCommand
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.generation.GenerationResult
import io.github.lxptechnologies.lxpmini.inference.CompletionRequest
import io.github.lxptechnologies.lxpmini.inference.CompletionResult
import io.github.lxptechnologies.lxpmini.inference.ContextOverflowPolicy
import io.github.lxptechnologies.lxpmini.inference.InferenceConcurrencyPolicy
import io.github.lxptechnologies.lxpmini.inference.InferenceMetrics
import io.github.lxptechnologies.lxpmini.inference.InferenceModelKind
import io.github.lxptechnologies.lxpmini.inference.InferenceModelMetadata
import io.github.lxptechnologies.lxpmini.inference.InferenceRuntimeLoader
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizerArtifactStore
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Path
import java.time.Duration

@ResourceLock(value = "DJL_ENGINE", mode = ResourceAccessMode.READ_WRITE)
class InferenceHttpServerTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    private val mapper = jacksonObjectMapper()
    private val client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build()

    @Test
    fun `HTTP contracts match the runtime reject invalid input and stream only when enabled`() {
        val artifacts = createArtifacts()
        val runtime = InferenceRuntimeLoader().load(MODEL_ID, artifacts.runDirectory, artifacts.tokenizerPath)
        val expected = runtime.complete(
            CompletionRequest(
                prompt = "abc",
                maxNewTokens = 3,
                sampling = SamplingOptions(strategy = SamplingStrategy.GREEDY),
                contextPolicy = ContextOverflowPolicy.REJECT,
            ),
        )
        val running = InferenceHttpServer().start(
            RuntimeInferenceService(runtime),
            InferenceServerOptions(port = 0, streamingEnabled = true),
        )
        try {
            val baseUrl = "http://${running.host}:${running.port}"
            val health = get("$baseUrl/health")
            assertThat(health.statusCode()).isEqualTo(200)
            val healthJson = mapper.readTree(health.body())
            assertThat(healthJson["status"].asText()).isEqualTo("ok")
            assertThat(healthJson["model"].asText()).isEqualTo(MODEL_ID)
            assertThat(healthJson["streaming_enabled"].asBoolean()).isTrue()
            assertThat(healthJson["device"].asText()).isEqualTo(runtime.metadata.selectedDevice)
            assertThat(healthJson["model_type"].asText()).isEqualTo("base")
            assertThat(healthJson["context_length"].asInt()).isEqualTo(8)

            val playground = get("$baseUrl/")
            assertThat(playground.statusCode()).isEqualTo(200)
            assertThat(playground.headers().firstValue("content-type").orElse("")).startsWith("text/html")
            assertThat(playground.body()).contains("LXP Mini", "pas encore un chatbot", "mémorise seulement")
            assertThat(get("$baseUrl/app.css").statusCode()).isEqualTo(200)
            val javascript = get("$baseUrl/app.js")
            assertThat(javascript.statusCode()).isEqualTo(200)
            assertThat(javascript.body()).contains("conversation.splice(0, conversation.length)")
            assertThat(javascript.body()).doesNotContain("/playground/clear")

            val unknown = get("$baseUrl/this-route-does-not-exist")
            assertError(unknown, 404, "not_found", null)

            val models = mapper.readTree(get("$baseUrl/v1/models").body())
            assertThat(models["object"].asText()).isEqualTo("list")
            assertThat(models["data"][0]["id"].asText()).isEqualTo(MODEL_ID)
            assertThat(models["data"][0]["object"].asText()).isEqualTo("model")
            assertThat(models["data"][0]["created"].asLong()).isPositive()
            assertThat(get("$baseUrl/v1/models/$MODEL_ID").statusCode()).isEqualTo(200)

            val completion = post(
                "$baseUrl/v1/completions",
                """{"model":"$MODEL_ID","prompt":"abc","max_tokens":3,"temperature":0,"seed":42}""",
            )
            assertThat(completion.statusCode()).isEqualTo(200)
            val completionJson = mapper.readTree(completion.body())
            assertThat(completionJson["id"].asText()).startsWith("cmpl-")
            assertThat(completionJson["object"].asText()).isEqualTo("text_completion")
            assertThat(completionJson["choices"][0]["text"].asText()).isEqualTo(expected.generatedText)
            assertThat(completionJson["choices"][0]["finish_reason"].asText()).isEqualTo("length")
            assertThat(completionJson["usage"]["prompt_tokens"].asInt()).isEqualTo(3)
            assertThat(completionJson["usage"]["completion_tokens"].asInt()).isEqualTo(3)
            assertThat(completionJson["usage"]["total_tokens"].asInt()).isEqualTo(6)

            assertError(
                post("$baseUrl/v1/completions", """{"model":"missing","prompt":"abc"}"""),
                404,
                "model_not_found",
                "model",
            )
            assertError(
                post(
                    "$baseUrl/v1/completions",
                    """{"model":"$MODEL_ID","prompt":"abc","max_tokens":6}""",
                ),
                400,
                "context_length_exceeded",
                "max_tokens",
            )
            assertError(
                post(
                    "$baseUrl/v1/completions",
                    """{"model":"$MODEL_ID","prompt":"abc","mystery":true}""",
                ),
                400,
                "unknown_parameter",
                "mystery",
            )
            assertError(
                post(
                    "$baseUrl/v1/completions",
                    """{"model":"$MODEL_ID","prompt":"abc","stop":"x"}""",
                ),
                400,
                "unsupported_feature",
                "stop",
            )

            val streamed = post(
                "$baseUrl/v1/completions",
                """{"model":"$MODEL_ID","prompt":"abc","max_tokens":3,"temperature":0,"stream":true}""",
            )
            assertThat(streamed.statusCode()).isEqualTo(200)
            assertThat(streamed.headers().firstValue("content-type").orElse("")).startsWith("text/event-stream")
            assertThat(streamed.body()).endsWith("data: [DONE]\n\n")
            assertThat(streamedText(streamed.body())).isEqualTo(expected.generatedText)
        } finally {
            running.close()
        }
        assertThat(runtime.isClosed).isTrue()

        val disabledRuntime = InferenceRuntimeLoader().load(MODEL_ID, artifacts.runDirectory, artifacts.tokenizerPath)
        InferenceHttpServer().start(
            RuntimeInferenceService(disabledRuntime),
            InferenceServerOptions(port = 0, streamingEnabled = false),
        ).use { disabled ->
            val response = post(
                "http://${disabled.host}:${disabled.port}/v1/completions",
                """{"model":"$MODEL_ID","prompt":"abc","stream":true}""",
            )
            assertError(response, 400, "unsupported_feature", "stream")
        }
        assertThat(disabledRuntime.isClosed).isTrue()
    }

    @Test
    fun `playground formats the conversation before calling inference`() {
        val inference = RecordingInferenceService()
        InferenceHttpServer().start(
            inference,
            InferenceServerOptions(port = 0),
        ).use { server ->
            val response = post(
                "http://${server.host}:${server.port}/playground/completions",
                """{
                    "system_prompt":"Be concise",
                    "messages":[
                        {"role":"user","content":"Hello"},
                        {"role":"assistant","content":"Hi"}
                    ],
                    "user_message":"Continue",
                    "max_new_tokens":2,
                    "top_p":1.0,
                    "top_k":0
                }""".trimIndent(),
            )

            assertThat(response.statusCode()).isEqualTo(200)
            val json = mapper.readTree(response.body())
            val expectedPrompt = "System: Be concise\nUser: Hello\nAssistant: Hi\nUser: Continue\nAssistant:"
            assertThat(json["formatted_prompt"].asText()).isEqualTo(expectedPrompt)
            assertThat(json["text"].asText()).isEqualTo("reply")
            assertThat(inference.lastRequest?.prompt).isEqualTo(expectedPrompt)
            assertThat(inference.lastRequest?.sampling?.strategy).isEqualTo(SamplingStrategy.GREEDY)
        }
        assertThat(inference.closed).isTrue()
    }

    private fun createArtifacts(): InferenceArtifacts {
        val runDirectory = temporaryDirectory.resolve("run")
        val tokenizerPath = temporaryDirectory.resolve("tokenizer.json")
        val command = CheckpointDemoCommand().apply {
            configPath = Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath()
            this.runDirectory = runDirectory
            beforeUpdates = 5
            afterUpdates = 1
        }
        check(command.call() == 0)
        ByteTokenizerArtifactStore().save(ByteTokenizer(), tokenizerPath)
        return InferenceArtifacts(runDirectory, tokenizerPath)
    }

    private fun get(url: String): HttpResponse<String> = client.send(
        HttpRequest.newBuilder(URI.create(url)).GET().timeout(Duration.ofSeconds(30)).build(),
        HttpResponse.BodyHandlers.ofString(),
    )

    private fun post(url: String, json: String): HttpResponse<String> = client.send(
        HttpRequest.newBuilder(URI.create(url))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .timeout(Duration.ofSeconds(30))
            .build(),
        HttpResponse.BodyHandlers.ofString(),
    )

    private fun assertError(response: HttpResponse<String>, status: Int, code: String, param: String?) {
        assertThat(response.statusCode()).isEqualTo(status)
        val error = mapper.readTree(response.body())["error"]
        assertThat(error["type"].asText()).isEqualTo("invalid_request_error")
        assertThat(error["code"].asText()).isEqualTo(code)
        if (param == null) assertThat(error["param"].isNull).isTrue() else assertThat(error["param"].asText()).isEqualTo(param)
        assertThat(error["message"].asText()).isNotBlank()
    }

    private fun streamedText(body: String): String = body.lineSequence()
        .filter { line -> line.startsWith("data: {") }
        .map { line -> mapper.readTree(line.removePrefix("data: ")) }
        .map { json -> json.getChoicesText() }
        .joinToString("")

    private fun JsonNode.getChoicesText(): String = get("choices")[0]["text"].asText()

    private data class InferenceArtifacts(val runDirectory: Path, val tokenizerPath: Path)

    private companion object {
        const val MODEL_ID = "lxp-mini-pr16-test-base"
    }
}

private class RecordingInferenceService : LocalInferenceService {
    override val metadata = InferenceModelMetadata(
        modelId = "playground-test-base",
        kind = InferenceModelKind.BASE,
        checkpointId = "step-1",
        checkpointSha256 = "0123456789abcdef",
        vocabularySize = 259,
        contextLength = 512,
        parameterCount = 6_752,
        tokenizerType = "byte",
        concurrencyPolicy = InferenceConcurrencyPolicy.SERIALIZED,
        createdAtEpochSeconds = 1,
        requestedDevice = "cpu",
        selectedDevice = "cpu",
        engineName = "PyTorch",
        nativeRuntimeVersion = "2.7.1",
        gpuCount = 0,
    )
    var lastRequest: CompletionRequest? = null
    var closed = false

    override fun countPromptTokens(prompt: String): Int = prompt.length

    override fun complete(request: CompletionRequest): CompletionResult {
        lastRequest = request
        val generation = GenerationResult(IntArray(request.prompt.length), intArrayOf(1, 2), false, emptyList())
        return CompletionResult(
            modelId = metadata.modelId,
            checkpointId = metadata.checkpointId,
            prompt = request.prompt,
            generatedText = "reply",
            completeText = request.prompt + "reply",
            promptTokens = request.prompt.length,
            generatedTokens = 2,
            stoppedByEos = false,
            generation = generation,
            metrics = InferenceMetrics(true, ContextOverflowPolicy.REJECT, 0, 0, 0, 0, 0, 0, 2, 0),
        )
    }

    override fun completeStreaming(
        request: CompletionRequest,
        onTextDelta: (String) -> Unit,
    ): CompletionResult = complete(request).also { onTextDelta(it.generatedText) }

    override fun close() {
        closed = true
    }
}
