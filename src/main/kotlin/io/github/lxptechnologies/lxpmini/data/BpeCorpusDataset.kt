package io.github.lxptechnologies.lxpmini.data

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizer
import java.nio.file.Path

class BpeCorpusDataset(
    trainCorpusPath: Path,
    validationCorpusPath: Path,
    tokenizer: BpeTokenizer,
    contextLength: Int,
    batchSize: Int,
    byteChunkSize: Int = StreamingBpeTokenReader.DEFAULT_BYTE_CHUNK_SIZE,
) {
    private val train = BpeCorpusPartition(trainCorpusPath, tokenizer, contextLength, batchSize, byteChunkSize)
    private val validation = BpeCorpusPartition(
        validationCorpusPath,
        tokenizer,
        contextLength,
        batchSize,
        byteChunkSize,
    )

    val trainTokenCount: Long = train.tokenCount
    val validationTokenCount: Long = validation.tokenCount
    val trainPlan: WindowPlan = train.plan
    val validationPlan: WindowPlan = validation.plan

    fun trainBatches(shuffleBufferSize: Int, seed: Long): TokenBatchReader {
        return train.batches(shuffleBufferSize, seed)
    }

    fun validationBatches(): TokenBatchReader = validation.batches()
}

class BpeCorpusPartition(
    corpusPath: Path,
    tokenizer: BpeTokenizer,
    private val contextLength: Int,
    private val batchSize: Int,
    byteChunkSize: Int = StreamingBpeTokenReader.DEFAULT_BYTE_CHUNK_SIZE,
) {
    private val readerFactory: TokenReaderFactory
    val tokenCount: Long
    val plan: WindowPlan

    init {
        if (contextLength <= 0) throw DatasetException("contextLength must be positive")
        if (batchSize <= 0) throw DatasetException("batchSize must be positive")
        if (byteChunkSize <= 0) throw DatasetException("byteChunkSize must be positive")
        readerFactory = TokenReaderFactory { StreamingBpeTokenReader(corpusPath, tokenizer, byteChunkSize) }
        tokenCount = CorpusTokenCounter.count(readerFactory)
        plan = WindowPlan(tokenCount, contextLength, contextLength)
    }

    fun batches(shuffleBufferSize: Int = 0, seed: Long = 0): TokenBatchReader {
        if (shuffleBufferSize < 0) throw DatasetException("shuffleBufferSize must be non-negative")
        var windows: WindowReader = SlidingWindowReader(readerFactory.open(), contextLength, contextLength)
        if (shuffleBufferSize > 0) windows = BufferedShufflingWindowReader(windows, shuffleBufferSize, seed)
        return TokenBatchReader(windows, batchSize)
    }
}

fun TokenBatchReader.asSequence(): Sequence<TokenBatch> = generateSequence(::readBatch)
