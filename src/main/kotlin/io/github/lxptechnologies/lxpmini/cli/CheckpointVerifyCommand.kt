package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.ndarray.NDManager
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointException
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.concurrent.Callable

@Command(
    name = "checkpoint-verify",
    mixinStandardHelpOptions = true,
    description = ["Verify checksums and load the latest checkpoint without continuing training."],
)
class CheckpointVerifyCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
) : Callable<Int> {
    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    override fun call(): Int = try {
        verify()
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    } catch (exception: CheckpointException) {
        System.err.println("Checkpoint error: ${exception.message}")
        2
    }

    private fun verify() {
        val configPath = runDirectory.resolve(RunStore.CONFIG_FILE)
        val config = configLoader.load(configPath)
        val configSha256 = Sha256.of(configPath)
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                val loaded = checkpointStore.loadLatest(runDirectory, model, manager, configSha256)
                println("Checkpoint directory:       ${loaded.directory.toAbsolutePath()}")
                println("Model SHA-256:              ${loaded.manifest.modelSha256}")
                println("Optimizer updates:          ${loaded.progress.optimizerUpdates}")
                println("Tokens seen:                ${loaded.progress.tokensSeen}")
                println("Model initialized:          ${model.isInitialized}")
                println("Optimizer counter restored: ${loaded.manifest.optimizerCounterRestored}")
                println("Scheduler restored:         ${loaded.manifest.schedulerRestored}")
                println("AdamW moments restored:     ${loaded.manifest.optimizerMomentsRestored}")
                println("Random state restored:      ${loaded.manifest.randomStateRestored}")
                println("Exact training resume:      ${loaded.manifest.exactTrainingResume}")
            }
        }
        println("Manager closed:             true")
    }
}
