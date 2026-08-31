package io.github.lxptechnologies.lxpmini.server

import org.springframework.boot.SpringBootConfiguration
import org.springframework.boot.WebApplicationType
import org.springframework.boot.autoconfigure.EnableAutoConfiguration
import org.springframework.boot.builder.SpringApplicationBuilder
import org.springframework.boot.web.context.WebServerApplicationContext
import org.springframework.context.ApplicationContextInitializer
import org.springframework.context.ApplicationListener
import org.springframework.context.ConfigurableApplicationContext
import org.springframework.context.annotation.Import
import org.springframework.context.event.ContextClosedEvent
import java.util.concurrent.CountDownLatch

@SpringBootConfiguration(proxyBeanMethods = false)
@EnableAutoConfiguration
@Import(OpenAiApiController::class, OpenAiExceptionHandler::class)
internal class InferenceServerApplication

data class InferenceServerOptions(
    val host: String = "127.0.0.1",
    val port: Int = 8080,
    val streamingEnabled: Boolean = false,
)

class InferenceHttpServer {
    fun start(
        service: LocalInferenceService,
        options: InferenceServerOptions,
    ): RunningInferenceServer {
        require(options.host.isNotBlank()) { "host cannot be blank" }
        require(options.port in 0..65_535) { "port must be in 0..65535" }
        val shutdown = CountDownLatch(1)
        val initializer = ApplicationContextInitializer<ConfigurableApplicationContext> { context ->
            context.beanFactory.registerSingleton("localInferenceService", service)
            context.beanFactory.registerSingleton("serverCapabilities", ServerCapabilities(options.streamingEnabled))
            context.addApplicationListener(
                ApplicationListener<ContextClosedEvent> {
                    try {
                        service.close()
                    } finally {
                        shutdown.countDown()
                    }
                },
            )
        }
        val context = try {
            SpringApplicationBuilder(InferenceServerApplication::class.java)
                .web(WebApplicationType.SERVLET)
                .initializers(initializer)
                .properties(serverProperties(options))
                .run()
        } catch (throwable: Throwable) {
            service.close()
            throw throwable
        }
        val webServer = context as? WebServerApplicationContext
            ?: run {
                context.close()
                throw IllegalStateException("Spring did not create a web server application context")
            }
        return RunningInferenceServer(context, shutdown, options.host, webServer.webServer.port, options.streamingEnabled)
    }

    private fun serverProperties(options: InferenceServerOptions): Map<String, Any> = mapOf(
        "server.address" to options.host,
        "server.port" to options.port,
        "server.shutdown" to "graceful",
        "spring.lifecycle.timeout-per-shutdown-phase" to "5s",
        "spring.main.banner-mode" to "off",
        "spring.jackson.deserialization.fail-on-unknown-properties" to true,
        "spring.web.resources.add-mappings" to false,
        "spring.mvc.throw-exception-if-no-handler-found" to true,
        "server.error.whitelabel.enabled" to false,
        "logging.level.root" to "WARN",
    )
}

class RunningInferenceServer internal constructor(
    private val context: ConfigurableApplicationContext,
    private val shutdown: CountDownLatch,
    val host: String,
    val port: Int,
    val streamingEnabled: Boolean,
) : AutoCloseable {
    val isRunning: Boolean
        get() = context.isRunning

    fun awaitShutdown() = shutdown.await()

    override fun close() = context.close()
}
