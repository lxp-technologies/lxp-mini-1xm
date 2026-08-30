package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointException
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.generation.AutoregressiveGenerator
import io.github.lxptechnologies.lxpmini.generation.GenerationException
import io.github.lxptechnologies.lxpmini.generation.GenerationStep
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.generation.TokenSampler
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import io.github.lxptechnologies.lxpmini.tokenizer.Tokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerArtifactLoader
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "generate",
    mixinStandardHelpOptions = true,
    description = ["Load a verified checkpoint and inspect autoregressive token generation."],
)
class GenerateCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
    private val tokenizerLoader: TokenizerArtifactLoader = TokenizerArtifactLoader(),
) : Callable<Int> {
    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--prompt"], required = true, paramLabel = "<text>")
    lateinit var prompt: String

    @Option(names = ["--max-new-tokens"], defaultValue = "16")
    var maxNewTokens: Int = 16

    @Option(names = ["--strategy"], defaultValue = "sample", description = ["greedy or sample"])
    lateinit var strategyName: String

    @Option(names = ["--temperature"], defaultValue = "1.0")
    var temperature: Double = 1.0

    @Option(names = ["--top-k"], defaultValue = "0", description = ["0 disables top-k filtering."])
    var topK: Int = 0

    @Option(names = ["--top-p"], defaultValue = "1.0")
    var topP: Double = 1.0

    @Option(names = ["--seed"], defaultValue = "42")
    var seed: Long = 42

    @Option(names = ["--add-bos"], description = ["Prefix the prompt with the BOS token."])
    var addBos: Boolean = false

    @Option(names = ["--show-candidates"], defaultValue = "5")
    var showCandidates: Int = 5

    override fun call(): Int = try {
        generate()
        0
    } catch (exception: ConfigException) {
        System.err.println("Configuration error: ${exception.message}")
        2
    } catch (exception: CheckpointException) {
        System.err.println("Checkpoint error: ${exception.message}")
        2
    } catch (exception: TokenizerException) {
        System.err.println("Tokenizer error: ${exception.message}")
        2
    } catch (exception: GenerationException) {
        System.err.println("Generation error: ${exception.message}")
        2
    } catch (exception: TensorShapeException) {
        System.err.println("Tensor shape error: ${exception.message}")
        2
    }

    private fun generate() {
        if (maxNewTokens <= 0) throw GenerationException("--max-new-tokens must be positive")
        if (showCandidates < 0) throw GenerationException("--show-candidates must be non-negative")
        val strategy = when (strategyName.lowercase(Locale.ROOT)) {
            "greedy" -> SamplingStrategy.GREEDY
            "sample" -> SamplingStrategy.SAMPLE
            else -> throw GenerationException("--strategy must be 'greedy' or 'sample'")
        }
        val configPath = runDirectory.resolve(RunStore.CONFIG_FILE)
        val config = configLoader.load(configPath)
        val tokenizerArtifact = tokenizerLoader.load(tokenizerPath)
        val tokenizer = tokenizerArtifact.tokenizer
        if (tokenizer.vocabularySize != config.model.vocabSize) {
            throw GenerationException(
                "Tokenizer vocabulary ${tokenizer.vocabularySize} does not match model vocabulary ${config.model.vocabSize}",
            )
        }
        val promptTokenIds = tokenizer.encode(prompt, addBos = addBos)
        if (promptTokenIds.isEmpty()) throw GenerationException("Prompt must produce at least one token; use text or --add-bos")
        val options = SamplingOptions(strategy, temperature, topK, topP)

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                val loaded = checkpointStore.loadLatest(runDirectory, model, manager, Sha256.of(configPath))
                val parameterStore = ParameterStore(manager, false)
                val generator = AutoregressiveGenerator(
                    config.model.contextLength,
                    config.model.vocabSize,
                    TokenSampler(seed),
                ) { context -> lastTokenLogits(model, parameterStore, manager, context) }
                val result = generator.generate(promptTokenIds, maxNewTokens, SpecialToken.EOS.id, options)

                println("Checkpoint:         ${loaded.directory.toAbsolutePath()}")
                println("Tokenizer:          ${tokenizerArtifact.type} (${tokenizer.vocabularySize} tokens)")
                println("Prompt:             ${prompt.quoted()}")
                println("Prompt token IDs:   ${promptTokenIds.display()}")
                println("Strategy:           ${strategy.name.lowercase(Locale.ROOT)}")
                println("Temperature:        $temperature")
                println("Top-k:              ${if (topK == 0) "disabled" else topK}")
                println("Top-p:              $topP")
                println("Seed:               $seed")
                println("Context length:     ${config.model.contextLength} (left sliding window)")
                result.steps.forEach { step -> printStep(step, tokenizer) }
                println("Generated token IDs:${result.generatedTokenIds.display(prefix = " ")}")
                println("Stopped by EOS:     ${result.stoppedByEos}")
                println("Generated text:     ${tokenizer.decode(result.generatedTokenIds).quoted()}")
                println("Complete text:      ${tokenizer.decode(result.allTokenIds).quoted()}")
            }
        }
        println("Manager closed:     true")
    }

    private fun lastTokenLogits(
        model: DecoderLanguageModel,
        parameterStore: ParameterStore,
        manager: NDManager,
        context: IntArray,
    ): FloatArray = manager.newSubManager().use { temporary ->
        val input = temporary.create(context.map(Int::toLong).toLongArray(), Shape(1, context.size.toLong()))
        val logits = model.forward(parameterStore, NDList(input), false).singletonOrThrow()
        logits.get("0, ${context.lastIndex}, :").toFloatArray()
    }

    private fun printStep(step: GenerationStep, tokenizer: Tokenizer) {
        val chosen = step.sampling.tokenId
        val candidates = step.sampling.candidates
            .take(showCandidates)
            .joinToString(", ") { candidate ->
                "${candidate.tokenId}:z=${candidate.scaledLogit.logit()},p=${candidate.probability.probability()}"
            }
        println(
            "step=%02d context=%s chosen=%d piece=%s z=%s p=%s candidates=[%s]".format(
                Locale.ROOT,
                step.number,
                step.contextTokenIds.display(),
                chosen,
                tokenPiece(tokenizer, chosen).quoted(),
                step.sampling.candidates.first { candidate -> candidate.tokenId == chosen }.scaledLogit.logit(),
                step.sampling.probability.probability(),
                candidates,
            ),
        )
    }

    private fun tokenPiece(tokenizer: Tokenizer, tokenId: Int): String =
        SpecialToken.fromId(tokenId)?.tokenText ?: tokenizer.decode(intArrayOf(tokenId))

    private fun String.escaped(): String = buildString {
        for (character in this@escaped) {
            when (character) {
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                else -> if (character.isISOControl()) append("\\u%04x".format(character.code)) else append(character)
            }
        }
    }

    private fun String.quoted(): String = "\"${escaped()}\""
    private fun IntArray.display(prefix: String = ""): String = prefix + joinToString(prefix = "[", postfix = "]")
    private fun Double.probability(): String = "%.6f".format(Locale.ROOT, this)
    private fun Double.logit(): String = "%.4f".format(Locale.ROOT, this)
}
