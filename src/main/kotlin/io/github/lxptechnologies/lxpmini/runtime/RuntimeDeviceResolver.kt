package io.github.lxptechnologies.lxpmini.runtime

import ai.djl.Device
import ai.djl.engine.Engine
import java.util.Locale

enum class RuntimeDeviceRequest(val value: String) {
    AUTO("auto"),
    CPU("cpu"),
    CUDA_0("cuda:0"),
    ;

    companion object {
        fun parse(value: String): RuntimeDeviceRequest = entries.firstOrNull {
            it.value == value.trim().lowercase(Locale.ROOT)
        } ?: throw RuntimeDeviceException(
            "runtime device must be one of: ${entries.joinToString { it.value }}; got '$value'",
        )
    }
}

data class RuntimeDeviceSelection(
    val requested: RuntimeDeviceRequest,
    val selected: Device,
    val selectedName: String,
    val engineName: String,
    val djlVersion: String,
    val nativeRuntimeVersion: String,
    val gpuCount: Int,
)

fun interface RuntimeDeviceSelector {
    fun resolve(requestedValue: String): RuntimeDeviceSelection
}

class RuntimeDeviceResolver(
    private val engineProvider: () -> RuntimeEngine = { DjlRuntimeEngine(Engine.getInstance()) },
) : RuntimeDeviceSelector {
    override fun resolve(requestedValue: String): RuntimeDeviceSelection {
        val requested = RuntimeDeviceRequest.parse(requestedValue)
        val engine = try {
            engineProvider()
        } catch (exception: Exception) {
            throw RuntimeDeviceException(
                "Cannot initialize the DJL runtime: ${exception.deepestMessage()}",
                exception,
            )
        }
        val gpuCount = try {
            engine.gpuCount()
        } catch (exception: Exception) {
            throw RuntimeDeviceException("Cannot inspect CUDA devices: ${exception.message}", exception)
        }
        val selected = when (requested) {
            RuntimeDeviceRequest.AUTO -> if (gpuCount > 0) Device.gpu(0) else Device.cpu()
            RuntimeDeviceRequest.CPU -> Device.cpu()
            RuntimeDeviceRequest.CUDA_0 -> {
                if (gpuCount < 1) {
                    throw RuntimeDeviceException(
                        "cuda:0 was explicitly requested, but ${engine.engineName()} reports no usable CUDA GPU",
                    )
                }
                Device.gpu(0)
            }
        }
        return RuntimeDeviceSelection(
            requested = requested,
            selected = selected,
            selectedName = if (selected.isGpu) "cuda:${selected.deviceId}" else "cpu",
            engineName = engine.engineName(),
            djlVersion = engine.djlVersion(),
            nativeRuntimeVersion = engine.nativeRuntimeVersion(),
            gpuCount = gpuCount,
        )
    }
}

interface RuntimeEngine {
    fun engineName(): String
    fun djlVersion(): String
    fun nativeRuntimeVersion(): String
    fun gpuCount(): Int
}

private class DjlRuntimeEngine(private val engine: Engine) : RuntimeEngine {
    override fun engineName(): String = engine.engineName
    override fun djlVersion(): String = Engine.getDjlVersion()
    override fun nativeRuntimeVersion(): String = engine.version
    override fun gpuCount(): Int = engine.gpuCount
}

class RuntimeDeviceException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

private fun Throwable.deepestMessage(): String = generateSequence(this) { throwable -> throwable.cause }
    .last()
    .message
    ?: message
    ?: this::class.java.simpleName
