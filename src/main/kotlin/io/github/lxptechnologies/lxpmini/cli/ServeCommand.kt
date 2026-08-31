package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.inference.InferenceException
import io.github.lxptechnologies.lxpmini.inference.InferenceRuntimeLoader
import io.github.lxptechnologies.lxpmini.server.InferenceHttpServer
import io.github.lxptechnologies.lxpmini.server.InferenceServerOptions
import io.github.lxptechnologies.lxpmini.server.RuntimeInferenceService
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.net.InetAddress
import java.nio.file.Path
import java.util.concurrent.Callable

@Command(
    name = "serve",
    mixinStandardHelpOptions = true,
    description = ["Start the local OpenAI-compatible completions server."],
)
class ServeCommand(
    private val runtimeLoader: InferenceRuntimeLoader = InferenceRuntimeLoader(),
    private val httpServer: InferenceHttpServer = InferenceHttpServer(),
) : Callable<Int> {
    @Option(names = ["--model-id"], defaultValue = "lxp-mini-1xm-base")
    lateinit var modelId: String

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--host"], defaultValue = "127.0.0.1")
    lateinit var host: String

    @Option(names = ["--port"], defaultValue = "8080")
    var port: Int = 8080

    @Option(
        names = ["--streaming-enabled"],
        negatable = true,
        defaultValue = "false",
        description = ["Allow stream=true SSE responses. Disabled by default."],
    )
    var streamingEnabled: Boolean = false

    @Option(
        names = ["--allow-remote"],
        description = ["Allow binding to a non-loopback address. The server has no authentication."],
    )
    var allowRemote: Boolean = false

    override fun call(): Int = try {
        validateNetworkOptions()
        val runtime = runtimeLoader.load(modelId, runDirectory, tokenizerPath)
        val service = RuntimeInferenceService(runtime)
        httpServer.start(service, InferenceServerOptions(host, port, streamingEnabled)).use { server ->
            println("Inference server:      http://${server.host}:${server.port}")
            println("Model:                 ${runtime.metadata.modelId}")
            println("Checkpoint:            ${runtime.metadata.checkpointId}")
            println("Streaming enabled:     ${server.streamingEnabled}")
            println("Concurrency:           ${runtime.metadata.concurrencyPolicy.name.lowercase()}")
            println("Press Ctrl+C to stop.")
            server.awaitShutdown()
        }
        0
    } catch (exception: InterruptedException) {
        Thread.currentThread().interrupt()
        0
    } catch (exception: InferenceException) {
        System.err.println("Inference server error: ${exception.message}")
        2
    } catch (exception: Exception) {
        System.err.println("Inference server error: ${exception.message}")
        2
    }

    private fun validateNetworkOptions() {
        if (port !in 1..65_535) throw InferenceException("--port must be in 1..65535")
        val address = try {
            InetAddress.getByName(host)
        } catch (exception: Exception) {
            throw InferenceException("--host cannot be resolved: $host", exception)
        }
        if (!address.isLoopbackAddress && !allowRemote) {
            throw InferenceException(
                "Refusing non-loopback host '$host' without --allow-remote; this server has no authentication",
            )
        }
    }
}
