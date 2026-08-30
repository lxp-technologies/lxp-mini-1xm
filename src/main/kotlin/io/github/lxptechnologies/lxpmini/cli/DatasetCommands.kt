package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.data.BufferedShufflingWindowReader
import io.github.lxptechnologies.lxpmini.data.CorpusTokenCounter
import io.github.lxptechnologies.lxpmini.data.DatasetException
import io.github.lxptechnologies.lxpmini.data.DeterministicTokenSplit
import io.github.lxptechnologies.lxpmini.data.IntArrayTokenReader
import io.github.lxptechnologies.lxpmini.data.RangedTokenReader
import io.github.lxptechnologies.lxpmini.data.SlidingWindowReader
import io.github.lxptechnologies.lxpmini.data.StreamingBpeTokenReader
import io.github.lxptechnologies.lxpmini.data.TokenBatch
import io.github.lxptechnologies.lxpmini.data.TokenBatchReader
import io.github.lxptechnologies.lxpmini.data.TokenRange
import io.github.lxptechnologies.lxpmini.data.TokenReaderFactory
import io.github.lxptechnologies.lxpmini.data.WindowPlan
import io.github.lxptechnologies.lxpmini.data.WindowReader
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.nio.file.Path
import java.util.concurrent.Callable

@Command(
    name = "dataset",
    description = ["Build and inspect next-token dataset windows."],
    subcommands = [DatasetWindowCommand::class, DatasetInspectCommand::class],
)
class DatasetCommand : Runnable {
    override fun run() {
        println("Choose a dataset command. Try: lxp-mini dataset inspect --help")
    }
}

@Command(
    name = "window",
    mixinStandardHelpOptions = true,
    description = ["Explain the one-token input/target shift using explicit token IDs."],
)
class DatasetWindowCommand : Callable<Int> {
    @Option(
        names = ["--tokens"],
        required = true,
        paramLabel = "<id,id,...>",
        description = ["Comma-separated token IDs."],
    )
    lateinit var tokenText: String

    @Option(names = ["--context-length"], required = true, paramLabel = "<count>")
    var contextLength: Int = 0

    @Option(names = ["--stride"], paramLabel = "<count>", description = ["Window step; defaults to context length."])
    var stride: Int? = null

    override fun call(): Int = try {
        val tokens = parseTokenIds(tokenText)
        val resolvedStride = stride ?: contextLength
        val plan = WindowPlan(tokens.size.toLong(), contextLength, resolvedStride)
        val reader = SlidingWindowReader(IntArrayTokenReader(tokens), contextLength, resolvedStride)
        val window = IntArray(contextLength + 1)
        var index = 0L

        println("Token stream:       ${tokens.contentToString()}")
        println("Window token shape: [T + 1] = [${contextLength + 1}]")
        println("Input shape:        [T] = [$contextLength]")
        println("Target shape:       [T] = [$contextLength]")
        reader.use {
            while (reader.readWindow(window)) {
                println("Window #${++index}")
                println("  input:  ${window.copyOfRange(0, contextLength).contentToString()}")
                println("  target: ${window.copyOfRange(1, contextLength + 1).contentToString()}")
            }
        }
        println("Complete windows:   ${plan.windowCount}")
        println("Trailing tokens:    ${plan.trailingTokenCount}")
        0
    } catch (exception: DatasetException) {
        System.err.println("Dataset error: ${exception.message}")
        2
    }

    private fun parseTokenIds(value: String): IntArray {
        if (value.isBlank()) throw DatasetException("--tokens must contain at least one token ID")
        return value.split(',').mapIndexed { index, part ->
            val tokenId = part.trim().toIntOrNull()
                ?: throw DatasetException("Token at position $index is not an integer: '${part.trim()}'")
            if (tokenId < 0) throw DatasetException("Token IDs must be non-negative")
            tokenId
        }.toIntArray()
    }
}

@Command(
    name = "inspect",
    mixinStandardHelpOptions = true,
    description = ["Stream a corpus into disjoint splits, next-token windows and primitive batches."],
)
class DatasetInspectCommand(
    private val artifactStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
) : Callable<Int> {
    @Option(names = ["--corpus"], required = true, paramLabel = "<file>")
    lateinit var corpusPath: Path

    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>")
    lateinit var tokenizerPath: Path

    @Option(names = ["--context-length"], required = true, paramLabel = "<count>")
    var contextLength: Int = 0

    @Option(names = ["--batch-size"], required = true, paramLabel = "<count>")
    var batchSize: Int = 0

    @Option(names = ["--validation-fraction"], defaultValue = "0.1", paramLabel = "<fraction>")
    var validationFraction: Double = 0.1

    @Option(names = ["--split"], defaultValue = "train", paramLabel = "<train|validation>")
    var selectedSplit: String = TRAIN_SPLIT

    @Option(names = ["--stride"], paramLabel = "<count>", description = ["Window step; defaults to context length."])
    var stride: Int? = null

    @Option(names = ["--shuffle-buffer"], defaultValue = "0", paramLabel = "<count>")
    var shuffleBufferSize: Int = 0

    @Option(names = ["--seed"], defaultValue = "42", paramLabel = "<long>")
    var seed: Long = 42

    @Option(names = ["--drop-last-batch"], description = ["Discard a final batch smaller than --batch-size."])
    var dropLastBatch: Boolean = false

    @Option(names = ["--show-batches"], defaultValue = "2", paramLabel = "<count>")
    var showBatches: Int = 2

    @Option(names = ["--byte-chunk-size"], defaultValue = "65536", hidden = true)
    var byteChunkSize: Int = StreamingBpeTokenReader.DEFAULT_BYTE_CHUNK_SIZE

    override fun call(): Int = try {
        validateOptions()
        val tokenizer = artifactStore.load(tokenizerPath).tokenizer
        val readerFactory = TokenReaderFactory {
            StreamingBpeTokenReader(corpusPath, tokenizer, byteChunkSize)
        }
        val totalTokenCount = CorpusTokenCounter.count(readerFactory)
        val split = DeterministicTokenSplit.contiguous(totalTokenCount, validationFraction)
        val selectedRange = selectRange(split.train, split.validation)
        val resolvedStride = stride ?: contextLength
        val plan = WindowPlan(selectedRange.size, contextLength, resolvedStride)

        printSummary(totalTokenCount, split.train, split.validation, selectedRange, resolvedStride, plan)
        if (showBatches > 0) inspectBatches(readerFactory, selectedRange, resolvedStride)
        0
    } catch (exception: DatasetException) {
        System.err.println("Dataset error: ${exception.message}")
        2
    } catch (exception: TokenizerException) {
        System.err.println("Tokenizer error: ${exception.message}")
        2
    }

    private fun validateOptions() {
        if (contextLength <= 0) throw DatasetException("--context-length must be positive")
        if (batchSize <= 0) throw DatasetException("--batch-size must be positive")
        if (shuffleBufferSize < 0) throw DatasetException("--shuffle-buffer must be non-negative")
        if (showBatches < 0) throw DatasetException("--show-batches must be non-negative")
        if (byteChunkSize <= 0) throw DatasetException("--byte-chunk-size must be positive")
        if (stride != null && requireNotNull(stride) <= 0) throw DatasetException("--stride must be positive")
        if (selectedSplit !in setOf(TRAIN_SPLIT, VALIDATION_SPLIT)) {
            throw DatasetException("--split must be '$TRAIN_SPLIT' or '$VALIDATION_SPLIT'")
        }
    }

    private fun selectRange(train: TokenRange, validation: TokenRange): TokenRange = when (selectedSplit) {
        TRAIN_SPLIT -> train
        VALIDATION_SPLIT -> validation
        else -> error("validated above")
    }

    private fun printSummary(
        totalTokenCount: Long,
        train: TokenRange,
        validation: TokenRange,
        selected: TokenRange,
        resolvedStride: Int,
        plan: WindowPlan,
    ) {
        val batchCount = if (dropLastBatch) {
            plan.windowCount / batchSize
        } else {
            (plan.windowCount + batchSize - 1) / batchSize
        }
        println("Corpus:             $corpusPath")
        println("Tokenizer:          $tokenizerPath")
        println("Total tokens:       $totalTokenCount")
        println("Train range:        [${train.startInclusive}, ${train.endExclusive}) = ${train.size} tokens")
        println("Validation range:   [${validation.startInclusive}, ${validation.endExclusive}) = ${validation.size} tokens")
        println("Selected split:     $selectedSplit [${selected.startInclusive}, ${selected.endExclusive})")
        println("Context length T:   $contextLength")
        println("Stride:             $resolvedStride")
        println("Complete windows:   ${plan.windowCount}")
        println("Trailing tokens:    ${plan.trailingTokenCount}")
        println("Planned batches:    $batchCount")
        println("Batch shape:        [B, T] = [$batchSize, $contextLength] (last B may be smaller)")
        println("Shuffle buffer:     $shuffleBufferSize")
        println("Seed:               $seed")
    }

    private fun inspectBatches(readerFactory: TokenReaderFactory, selectedRange: TokenRange, resolvedStride: Int) {
        val rangedReader = RangedTokenReader(readerFactory.open(), selectedRange)
        var windows: WindowReader = SlidingWindowReader(rangedReader, contextLength, resolvedStride)
        if (shuffleBufferSize > 0) {
            windows = BufferedShufflingWindowReader(windows, shuffleBufferSize, seed)
        }

        TokenBatchReader(windows, batchSize, dropLastBatch).use { batches ->
            repeat(showBatches) { batchIndex ->
                val batch = batches.readBatch() ?: return
                printBatch(batchIndex + 1, batch)
            }
        }
    }

    private fun printBatch(index: Int, batch: TokenBatch) {
        println("Batch #$index actual shape [${batch.batchSize}, ${batch.sequenceLength}]")
        for (row in 0 until batch.batchSize) {
            println("  row $row input:  ${batch.inputRow(row).contentToString()}")
            println("  row $row target: ${batch.targetRow(row).contentToString()}")
        }
    }

    private companion object {
        const val TRAIN_SPLIT = "train"
        const val VALIDATION_SPLIT = "validation"
    }
}
