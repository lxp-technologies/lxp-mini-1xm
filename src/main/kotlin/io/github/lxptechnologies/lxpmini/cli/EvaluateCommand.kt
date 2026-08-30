package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.ndarray.NDManager
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointException
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.data.BpeCorpusPartition
import io.github.lxptechnologies.lxpmini.data.DatasetException
import io.github.lxptechnologies.lxpmini.data.StreamingBpeTokenReader
import io.github.lxptechnologies.lxpmini.data.asSequence
import io.github.lxptechnologies.lxpmini.evaluation.EvaluationException
import io.github.lxptechnologies.lxpmini.evaluation.LanguageModelEvaluator
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "evaluate",
    mixinStandardHelpOptions = true,
    description = ["Compare checkpoint loss, perplexity and throughput on the verified validation corpus."],
)
class EvaluateCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
    private val tokenizerStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
    private val runStore: RunStore = RunStore(),
) : Callable<Int> {
    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--validation-corpus"], required = true, paramLabel = "<file>")
    lateinit var validationCorpusPath: Path

    @Option(names = ["--tokenizer"], paramLabel = "<file>", description = ["Defaults to <run-dir>/tokenizer.json."])
    var tokenizerPath: Path? = null

    @Option(
        names = ["--checkpoint"],
        paramLabel = "<step-id>",
        description = ["Repeat to compare checkpoints; defaults to latest."],
    )
    var checkpointIds: Array<String> = emptyArray()

    @Option(names = ["--batch-size"], defaultValue = "0", description = ["0 uses training.batchSize."])
    var requestedBatchSize: Int = 0

    @Option(names = ["--max-batches"], defaultValue = "0", description = ["0 evaluates the complete split."])
    var maxBatches: Int = 0

    @Option(names = ["--byte-chunk-size"], defaultValue = "65536", hidden = true)
    var byteChunkSize: Int = StreamingBpeTokenReader.DEFAULT_BYTE_CHUNK_SIZE

    override fun call(): Int = try {
        evaluateCheckpoints()
        0
    } catch (exception: ConfigException) {
        fail("Configuration", exception)
    } catch (exception: CheckpointException) {
        fail("Checkpoint", exception)
    } catch (exception: TokenizerException) {
        fail("Tokenizer", exception)
    } catch (exception: DatasetException) {
        fail("Dataset", exception)
    } catch (exception: EvaluationException) {
        fail("Evaluation", exception)
    } catch (exception: TensorShapeException) {
        fail("Tensor shape", exception)
    }

    private fun evaluateCheckpoints() {
        validateOptions()
        val configPath = runDirectory.resolve(RunStore.CONFIG_FILE)
        val config = configLoader.load(configPath)
        val metadata = runStore.loadMetadata(runDirectory)
        val resolvedTokenizerPath = tokenizerPath ?: runDirectory.resolve(RunStore.TOKENIZER_FILE)
        val corpusSha256 = Sha256.of(validationCorpusPath)
        if (metadata.validationDatasetSha256 == null) {
            throw EvaluationException("Run does not declare a separate validation dataset checksum")
        }
        if (corpusSha256 != metadata.validationDatasetSha256) {
            throw EvaluationException(
                "Validation checksum $corpusSha256 does not match run dataset ${metadata.validationDatasetSha256}",
            )
        }
        val tokenizerSha256 = Sha256.of(resolvedTokenizerPath)
        if (metadata.tokenizerSha256 != null && tokenizerSha256 != metadata.tokenizerSha256) {
            throw EvaluationException(
                "Tokenizer checksum $tokenizerSha256 does not match run tokenizer ${metadata.tokenizerSha256}",
            )
        }
        val tokenizer = tokenizerStore.load(resolvedTokenizerPath).tokenizer
        if (tokenizer.vocabularySize != config.model.vocabSize) {
            throw EvaluationException(
                "Tokenizer vocabulary ${tokenizer.vocabularySize} does not match model vocabulary ${config.model.vocabSize}",
            )
        }
        val batchSize = if (requestedBatchSize == 0) config.training.batchSize else requestedBatchSize
        val validation = BpeCorpusPartition(
            validationCorpusPath,
            tokenizer,
            config.model.contextLength,
            batchSize,
            byteChunkSize,
        )
        if (validation.plan.windowCount == 0L) {
            throw EvaluationException("Validation corpus has no complete context window")
        }

        println("Corpus SHA-256:     $corpusSha256 (verified)")
        println("Tokenizer SHA-256:  $tokenizerSha256 (verified)")
        println("Validation tokens:  ${validation.tokenCount}")
        println("Validation windows: ${validation.plan.windowCount}")
        println("checkpoint       update       loss perplexity      tokens/s batches tokens")

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                val ids = checkpointIds.ifEmpty { arrayOf(LATEST) }
                for (id in ids) {
                    val loaded = if (id == LATEST) {
                        checkpointStore.loadLatest(runDirectory, model, manager, Sha256.of(configPath))
                    } else {
                        checkpointStore.load(
                            runDirectory.resolve(CheckpointStore.CHECKPOINTS_DIRECTORY).resolve(id),
                            model,
                            manager,
                            Sha256.of(configPath),
                        )
                    }
                    val metrics = validation.batches().use { batches ->
                        LanguageModelEvaluator(model, manager).evaluate(
                            batches.asSequence(),
                            if (maxBatches == 0) Int.MAX_VALUE else maxBatches,
                        )
                    }
                    println(
                        "%-16s %6d %10.6f %10.4f %13.2f %7d %6d".format(
                            Locale.ROOT,
                            loaded.manifest.checkpointId,
                            loaded.progress.optimizerUpdates,
                            metrics.loss,
                            metrics.perplexity,
                            metrics.tokensPerSecond,
                            metrics.batchCount,
                            metrics.tokenCount,
                        ),
                    )
                }
            }
        }
        println("Gradients computed: false")
    }

    private fun validateOptions() {
        if (requestedBatchSize < 0) throw EvaluationException("--batch-size must be non-negative")
        if (maxBatches < 0) throw EvaluationException("--max-batches must be non-negative")
        if (byteChunkSize <= 0) throw EvaluationException("--byte-chunk-size must be positive")
        checkpointIds.firstOrNull { id -> !id.matches(CHECKPOINT_ID_PATTERN) }?.let { id ->
            throw EvaluationException("Invalid checkpoint ID '$id'; expected step-00000000")
        }
    }

    private fun fail(kind: String, exception: Exception): Int {
        System.err.println("$kind error: ${exception.message}")
        return 2
    }

    private companion object {
        const val LATEST = "latest"
        val CHECKPOINT_ID_PATTERN = Regex("step-[0-9]{8}")
    }
}
