package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.model.ParameterCounter
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.concurrent.Callable

@Command(
    name = "lxp-mini",
    mixinStandardHelpOptions = true,
    description = ["Explore and build a small language model, one concept at a time."],
    subcommands = [
        ModelCommand::class,
        TokenizerCommand::class,
        DatasetCommand::class,
        TrainCommand::class,
        GenerateCommand::class,
        InferenceCommand::class,
        EvaluateCommand::class,
        ExperimentCommand::class,
    ],
)
class LxpMiniCommand : Runnable {
    override fun run() {
        println("Choose a command. Try: lxp-mini --help")
    }
}

@Command(
    name = "train",
    description = ["Run focused training experiments."],
    subcommands = [
        OverfitBatchCommand::class,
        CheckpointDemoCommand::class,
        CheckpointVerifyCommand::class,
        CorpusTrainingCommand::class,
    ],
)
class TrainCommand : Runnable {
    override fun run() {
        println("Choose a training command. Try: lxp-mini train overfit-batch --help")
    }
}

@Command(
    name = "model",
    description = ["Inspect model configurations."],
    subcommands = [
        ModelInfoCommand::class,
        ModelComponentsCommand::class,
        ModelAttentionCommand::class,
        ModelBlockCommand::class,
        ModelForwardCommand::class,
    ],
)
class ModelCommand : Runnable {
    override fun run() {
        println("Choose a model command. Try: lxp-mini model info --help")
    }
}

@Command(
    name = "info",
    mixinStandardHelpOptions = true,
    description = ["Validate a YAML configuration and explain its parameter count."],
)
class ModelInfoCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val parameterCounter: ParameterCounter = ParameterCounter(),
) : Callable<Int> {
    @Option(
        names = ["--config"],
        required = true,
        paramLabel = "<file>",
        description = ["Path to the model YAML configuration."],
    )
    lateinit var configPath: Path

    override fun call(): Int = try {
        val config = configLoader.load(configPath)
        val count = parameterCounter.count(config.model)
        println(count.format(config.model))
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    }
}
