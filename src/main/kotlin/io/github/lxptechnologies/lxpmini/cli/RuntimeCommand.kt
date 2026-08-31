package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.runtime.RuntimeDeviceException
import io.github.lxptechnologies.lxpmini.runtime.RuntimeDeviceResolver
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.util.concurrent.Callable

@Command(
    name = "runtime",
    description = ["Inspect the local DJL runtime and device selection."],
    subcommands = [RuntimeInfoCommand::class],
)
class RuntimeCommand : Runnable {
    override fun run() {
        println("Choose a runtime command. Try: lxp-mini runtime info --help")
    }
}

@Command(
    name = "info",
    mixinStandardHelpOptions = true,
    description = ["Show the engine, native runtime and selected execution device."],
)
class RuntimeInfoCommand(
    private val resolver: RuntimeDeviceResolver = RuntimeDeviceResolver(),
) : Callable<Int> {
    @Option(names = ["--device"], defaultValue = "auto", description = ["auto, cpu, or cuda:0"])
    lateinit var requestedDevice: String

    override fun call(): Int = try {
        val selection = resolver.resolve(requestedDevice)
        println("DJL version:            ${selection.djlVersion}")
        println("Engine:                 ${selection.engineName}")
        println("Native runtime:         ${selection.nativeRuntimeVersion}")
        println("GPU count:              ${selection.gpuCount}")
        println("Requested device:       ${selection.requested.value}")
        println("Selected device:        ${selection.selectedName}")
        0
    } catch (exception: RuntimeDeviceException) {
        System.err.println("Runtime device error: ${exception.message}")
        2
    }
}
