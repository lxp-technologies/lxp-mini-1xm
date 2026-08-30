package io.github.lxptechnologies.lxpmini.cli

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.module.kotlin.KotlinModule
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointException
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunEnvironment
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.checkpoint.TrainingMetricRecord
import io.github.lxptechnologies.lxpmini.config.ConfigException
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.data.BpeCorpusDataset
import io.github.lxptechnologies.lxpmini.data.DatasetException
import io.github.lxptechnologies.lxpmini.data.StreamingBpeTokenReader
import io.github.lxptechnologies.lxpmini.data.TokenBatch
import io.github.lxptechnologies.lxpmini.data.TokenBatchReader
import io.github.lxptechnologies.lxpmini.data.asSequence
import io.github.lxptechnologies.lxpmini.evaluation.EvaluationException
import io.github.lxptechnologies.lxpmini.evaluation.EvaluationMetrics
import io.github.lxptechnologies.lxpmini.evaluation.LanguageModelEvaluator
import io.github.lxptechnologies.lxpmini.generation.AutoregressiveGenerator
import io.github.lxptechnologies.lxpmini.generation.GenerationException
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.generation.TokenSampler
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.model.TensorShapeException
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import io.github.lxptechnologies.lxpmini.training.LanguageModelTrainer
import io.github.lxptechnologies.lxpmini.training.OptimizerUpdateMetrics
import io.github.lxptechnologies.lxpmini.training.TrainingException
import io.github.lxptechnologies.lxpmini.training.TrainingProgress
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "corpus",
    mixinStandardHelpOptions = true,
    description = ["Train a fresh measured run on a streaming BPE corpus with periodic validation."],
)
class CorpusTrainingCommand(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val tokenizerStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
    private val runStore: RunStore = RunStore(),
    private val checkpointStore: CheckpointStore = CheckpointStore(),
) : Callable<Int> {
    @Option(names = ["--config"], required = true, paramLabel = "<file>")
    lateinit var configPath: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--train-corpus"], required = true, paramLabel = "<file>")
    lateinit var trainCorpusPath: Path

    @Option(names = ["--validation-corpus"], required = true, paramLabel = "<file>")
    lateinit var validationCorpusPath: Path

    @Option(names = ["--run-dir"], required = true, paramLabel = "<directory>")
    lateinit var runDirectory: Path

    @Option(names = ["--updates"], defaultValue = "40")
    var updates: Int = 40

    @Option(names = ["--eval-every"], defaultValue = "10")
    var evaluateEvery: Int = 10

    @Option(names = ["--checkpoint-every"], defaultValue = "20")
    var checkpointEvery: Int = 20

    @Option(names = ["--shuffle-buffer"], defaultValue = "32")
    var shuffleBufferSize: Int = 32

    @Option(names = ["--max-validation-batches"], defaultValue = "0", description = ["0 evaluates all batches."])
    var maxValidationBatches: Int = 0

    @Option(names = ["--prompt"], paramLabel = "<text>", description = ["Repeat for fixed sample prompts."])
    var prompts: Array<String> = emptyArray()

    @Option(names = ["--sample-tokens"], defaultValue = "16")
    var sampleTokens: Int = 16

    @Option(names = ["--byte-chunk-size"], defaultValue = "65536", hidden = true)
    var byteChunkSize: Int = StreamingBpeTokenReader.DEFAULT_BYTE_CHUNK_SIZE

    override fun call(): Int = try {
        trainCorpus()
        0
    } catch (exception: ConfigException) {
        fail("Configuration", exception)
    } catch (exception: TokenizerException) {
        fail("Tokenizer", exception)
    } catch (exception: DatasetException) {
        fail("Dataset", exception)
    } catch (exception: TrainingException) {
        fail("Training", exception)
    } catch (exception: EvaluationException) {
        fail("Evaluation", exception)
    } catch (exception: GenerationException) {
        fail("Generation", exception)
    } catch (exception: CheckpointException) {
        fail("Checkpoint", exception)
    } catch (exception: TensorShapeException) {
        fail("Tensor shape", exception)
    } catch (exception: IOException) {
        fail("I/O", exception)
    }

    private fun trainCorpus() {
        val config = configLoader.load(configPath)
        validateOptions(config.training.warmupSteps)
        val trainedTokenizer = tokenizerStore.load(tokenizerPath)
        val tokenizer = trainedTokenizer.tokenizer
        val trainSha256 = Sha256.of(trainCorpusPath)
        if (trainedTokenizer.metadata.corpusSha256 != trainSha256) {
            throw TrainingException(
                "Tokenizer was trained on ${trainedTokenizer.metadata.corpusSha256}, not train corpus $trainSha256",
            )
        }
        if (tokenizer.vocabularySize != config.model.vocabSize) {
            throw TrainingException(
                "Tokenizer vocabulary ${tokenizer.vocabularySize} does not match model vocabulary ${config.model.vocabSize}",
            )
        }
        val dataset = BpeCorpusDataset(
            trainCorpusPath,
            validationCorpusPath,
            tokenizer,
            config.model.contextLength,
            config.training.batchSize,
            byteChunkSize,
        )
        requireUsableSplits(dataset)

        val engine = Engine.getInstance()
        val engineSeed = config.training.seed.toInt()
        if (engineSeed.toLong() != config.training.seed) {
            throw TrainingException("training.seed must fit in a 32-bit integer for DJL")
        }
        engine.setRandomSeed(engineSeed)
        val initializedRun = runStore.initialize(
            runDirectory,
            configPath,
            RunEnvironment(engine.engineName, engine.version, Device.cpu().toString()),
            trainSha256,
            config.training.seed,
            datasetKind = "utf8-explicit-train-validation-files",
            tokenizer = "byte-bpe",
            tokenizerSha256 = Sha256.of(tokenizerPath),
            validationDatasetSha256 = Sha256.of(validationCorpusPath),
        )
        Files.copy(
            tokenizerPath,
            runDirectory.resolve(RunStore.TOKENIZER_FILE),
            StandardCopyOption.REPLACE_EXISTING,
        )
        writeExperimentManifest(dataset)
        printRunSummary(dataset, tokenizer, initializedRun.configSha256)

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                val shape = Shape(config.training.batchSize.toLong(), config.model.contextLength.toLong())
                model.initialize(manager, DataType.FLOAT32, shape)
                val trainer = LanguageModelTrainer(model, manager, config.training, updates)
                val evaluator = LanguageModelEvaluator(model, manager)
                val startedAt = System.nanoTime()
                var epoch = 0
                var trainBatches = dataset.trainBatches(shuffleBufferSize, epochSeed(config.training.seed, epoch))

                fun nextBatch(): TokenBatch {
                    var batch = trainBatches.readBatch()
                    while (batch == null) {
                        trainBatches.close()
                        epoch += 1
                        trainBatches = dataset.trainBatches(shuffleBufferSize, epochSeed(config.training.seed, epoch))
                        batch = trainBatches.readBatch()
                    }
                    return batch
                }

                try {
                    repeat(updates) {
                        var weightedTrainLoss = 0.0
                        var trainTokens = 0L
                        var updateMetrics: OptimizerUpdateMetrics? = null
                        repeat(config.training.gradientAccumulationSteps) {
                            val batch = nextBatch()
                            manager.newSubManager().use { temporary ->
                                val batchShape = Shape(batch.batchSize.toLong(), batch.sequenceLength.toLong())
                                val metrics = trainer.trainMicroBatch(
                                    temporary.create(batch.inputIds.toLongArray(), batchShape),
                                    temporary.create(batch.targetIds.toLongArray(), batchShape),
                                )
                                weightedTrainLoss += metrics.loss * batch.inputIds.size
                                trainTokens += batch.inputIds.size
                                updateMetrics = metrics.optimizerUpdate ?: updateMetrics
                            }
                        }
                        val completed = requireNotNull(updateMetrics)
                        val trainLoss = (weightedTrainLoss / trainTokens).toFloat()
                        val shouldEvaluate = completed.updateNumber == 1 ||
                            completed.updateNumber % evaluateEvery == 0 || completed.updateNumber == updates
                        val validation = if (shouldEvaluate) evaluate(dataset, evaluator) else null
                        val shouldCheckpoint = completed.updateNumber % checkpointEvery == 0 || completed.updateNumber == updates
                        val checkpointId = if (shouldCheckpoint) {
                            checkpointStore.save(
                                runDirectory,
                                model,
                                TrainingProgress(trainer.optimizerUpdates, trainer.tokensSeen),
                                updates,
                                initializedRun.configSha256,
                            ).manifest.checkpointId
                        } else {
                            null
                        }
                        if (shouldEvaluate) writeSamples(model, manager, tokenizer, completed.updateNumber)

                        val elapsedSeconds = (System.nanoTime() - startedAt).coerceAtLeast(1L) / NANOS_PER_SECOND
                        val throughput = trainer.tokensSeen / elapsedSeconds
                        runStore.appendMetric(
                            runDirectory,
                            TrainingMetricRecord(
                                phase = "train",
                                update = completed.updateNumber,
                                tokensSeen = trainer.tokensSeen,
                                loss = trainLoss,
                                learningRate = completed.learningRate,
                                gradientNorm = completed.gradientNormBeforeClip,
                                clipped = completed.clipped,
                                validationLoss = validation?.loss,
                                validationPerplexity = validation?.perplexity,
                                tokensPerSecond = throughput,
                                elapsedSeconds = elapsedSeconds,
                            ),
                        )
                        if (shouldEvaluate || shouldCheckpoint) {
                            printProgress(trainLoss, validation, completed, throughput, epoch, checkpointId)
                        }
                    }
                } finally {
                    trainBatches.close()
                }
            }
        }
        println("Run complete:       ${runDirectory.toAbsolutePath()}")
        println("Exact resume:       false (AdamW moments and RNG are not serialized)")
        println("Managers closed:    true")
    }

    private fun evaluate(dataset: BpeCorpusDataset, evaluator: LanguageModelEvaluator): EvaluationMetrics =
        dataset.validationBatches().use { batches ->
            evaluator.evaluate(
                batches.asSequence(),
                if (maxValidationBatches == 0) Int.MAX_VALUE else maxValidationBatches,
            )
        }

    private fun writeSamples(
        model: DecoderLanguageModel,
        manager: NDManager,
        tokenizer: BpeTokenizer,
        update: Int,
    ) {
        val parameterStore = ParameterStore(manager, false)
        val resolvedPrompts = prompts.ifEmpty { DEFAULT_PROMPTS }
        val lines = resolvedPrompts.mapIndexed { index, prompt ->
            val promptIds = tokenizer.encode(prompt)
            val generator = AutoregressiveGenerator(
                model.config.contextLength,
                model.config.vocabSize,
                TokenSampler(SAMPLE_SEED + index),
            ) { context ->
                manager.newSubManager().use { temporary ->
                    val input = temporary.create(context.toLongArray(), Shape(1, context.size.toLong()))
                    val logits = model.forward(parameterStore, NDList(input), false).singletonOrThrow()
                    logits.get("0, ${context.lastIndex}, :").toFloatArray()
                }
            }
            val result = generator.generate(
                promptIds,
                sampleTokens,
                SpecialToken.EOS.id,
                SamplingOptions(strategy = SamplingStrategy.GREEDY),
            )
            "prompt=${prompt.printable()}\ttext=${decodeSample(tokenizer, result.allTokenIds)}"
        }
        val samplePath = runDirectory.resolve(RunStore.SAMPLES_DIRECTORY).resolve(
            "step-%08d.txt".format(update),
        )
        Files.write(samplePath, lines, StandardCharsets.UTF_8)
    }

    private fun requireUsableSplits(dataset: BpeCorpusDataset) {
        if (dataset.trainPlan.windowCount == 0L) throw TrainingException("Train split has no complete context window")
        if (dataset.validationPlan.windowCount == 0L) {
            throw TrainingException("Validation split has no complete context window")
        }
    }

    private fun writeExperimentManifest(dataset: BpeCorpusDataset) {
        val manifest = CorpusExperimentManifest(
            updates = updates,
            evaluateEvery = evaluateEvery,
            checkpointEvery = checkpointEvery,
            shuffleBufferSize = shuffleBufferSize,
            maxValidationBatches = maxValidationBatches,
            prompts = prompts.ifEmpty { DEFAULT_PROMPTS }.toList(),
            sampleTokens = sampleTokens,
            trainTokenCount = dataset.trainTokenCount,
            validationTokenCount = dataset.validationTokenCount,
            trainWindowCount = dataset.trainPlan.windowCount,
            validationWindowCount = dataset.validationPlan.windowCount,
        )
        val mapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
            .enable(SerializationFeature.INDENT_OUTPUT)
        Files.newBufferedWriter(runDirectory.resolve(RunStore.EXPERIMENT_FILE)).use { writer ->
            mapper.writeValue(writer, manifest)
        }
    }

    private fun validateOptions(warmupSteps: Int) {
        if (updates <= warmupSteps) throw TrainingException("--updates must be greater than warmupSteps ($warmupSteps)")
        if (evaluateEvery <= 0) throw TrainingException("--eval-every must be positive")
        if (checkpointEvery <= 0) throw TrainingException("--checkpoint-every must be positive")
        if (shuffleBufferSize < 0) throw TrainingException("--shuffle-buffer must be non-negative")
        if (maxValidationBatches < 0) throw TrainingException("--max-validation-batches must be non-negative")
        if (sampleTokens <= 0) throw TrainingException("--sample-tokens must be positive")
        if (byteChunkSize <= 0) throw TrainingException("--byte-chunk-size must be positive")
        prompts.firstOrNull(String::isBlank)?.let { throw TrainingException("--prompt cannot be blank") }
    }

    private fun printRunSummary(dataset: BpeCorpusDataset, tokenizer: BpeTokenizer, configSha256: String) {
        println("Train SHA-256:       ${Sha256.of(trainCorpusPath)}")
        println("Validation SHA-256:  ${Sha256.of(validationCorpusPath)}")
        println("Tokenizer SHA-256:   ${Sha256.of(tokenizerPath)}")
        println("Config SHA-256:      $configSha256")
        println("Vocabulary:          ${tokenizer.vocabularySize}")
        println("Total tokens:        ${dataset.trainTokenCount + dataset.validationTokenCount}")
        println("Train tokens/windows:${dataset.trainTokenCount}/${dataset.trainPlan.windowCount}")
        println("Validation tokens/windows: ${dataset.validationTokenCount}/${dataset.validationPlan.windowCount}")
        println("Updates:             $updates")
    }

    private fun printProgress(
        trainLoss: Float,
        validation: EvaluationMetrics?,
        metrics: OptimizerUpdateMetrics,
        throughput: Double,
        epoch: Int,
        checkpointId: String?,
    ) {
        println(
            "update=%4d epoch=%3d train=%s validation=%s perplexity=%s tokens/s=%8.2f checkpoint=%s".format(
                Locale.ROOT,
                metrics.updateNumber,
                epoch + 1,
                trainLoss.decimal(),
                validation?.loss?.decimal() ?: "-",
                validation?.perplexity?.decimal() ?: "-",
                throughput,
                checkpointId ?: "-",
            ),
        )
    }

    private fun epochSeed(seed: Long, epoch: Int): Long = seed + epoch
    private fun decodeSample(tokenizer: BpeTokenizer, tokenIds: IntArray): String = try {
        tokenizer.decode(tokenIds).printable()
    } catch (_: TokenizerException) {
        "<invalid UTF-8; tokenIds=${tokenIds.contentToString()}>"
    }

    private fun Float.decimal(): String = "%.6f".format(Locale.ROOT, this)
    private fun Double.decimal(): String = "%.6f".format(Locale.ROOT, this)
    private fun String.printable(): String = replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    private fun IntArray.toLongArray(): LongArray = LongArray(size) { index -> this[index].toLong() }

    private fun fail(kind: String, exception: Exception): Int {
        System.err.println("$kind error: ${exception.message}")
        return 2
    }

    private companion object {
        const val NANOS_PER_SECOND = 1_000_000_000.0
        const val SAMPLE_SEED = 42L
        val DEFAULT_PROMPTS = arrayOf("Bonjour", "Le modèle")
    }
}

@JsonPropertyOrder(
    "schemaVersion",
    "updates",
    "evaluateEvery",
    "checkpointEvery",
    "shuffleBufferSize",
    "maxValidationBatches",
    "prompts",
    "sampleTokens",
    "trainTokenCount",
    "validationTokenCount",
    "trainWindowCount",
    "validationWindowCount",
)
private data class CorpusExperimentManifest(
    val schemaVersion: Int = 1,
    val updates: Int,
    val evaluateEvery: Int,
    val checkpointEvery: Int,
    val shuffleBufferSize: Int,
    val maxValidationBatches: Int,
    val prompts: List<String>,
    val sampleTokens: Int,
    val trainTokenCount: Long,
    val validationTokenCount: Long,
    val trainWindowCount: Long,
    val validationWindowCount: Long,
)
