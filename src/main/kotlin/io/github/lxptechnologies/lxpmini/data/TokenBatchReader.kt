package io.github.lxptechnologies.lxpmini.data

data class TokenBatch(
    val batchSize: Int,
    val sequenceLength: Int,
    val inputIds: IntArray,
    val targetIds: IntArray,
) {
    init {
        if (batchSize <= 0 || sequenceLength <= 0) throw DatasetException("Batch dimensions must be positive")
        val expectedSize = batchSize.toLong() * sequenceLength
        if (expectedSize > Int.MAX_VALUE) throw DatasetException("Batch contains too many token positions")
        if (inputIds.size.toLong() != expectedSize || targetIds.size.toLong() != expectedSize) {
            throw DatasetException("Batch arrays must have shape [$batchSize, $sequenceLength]")
        }
    }

    fun inputRow(index: Int): IntArray = row(inputIds, index)

    fun targetRow(index: Int): IntArray = row(targetIds, index)

    private fun row(values: IntArray, index: Int): IntArray {
        if (index !in 0 until batchSize) throw DatasetException("Batch row $index is outside 0..${batchSize - 1}")
        val start = index * sequenceLength
        return values.copyOfRange(start, start + sequenceLength)
    }
}

class TokenBatchReader(
    private val windowReader: WindowReader,
    private val requestedBatchSize: Int,
    private val dropLastBatch: Boolean = false,
) : AutoCloseable {
    init {
        if (requestedBatchSize <= 0) throw DatasetException("batchSize must be positive")
        if (requestedBatchSize.toLong() * windowReader.contextLength > Int.MAX_VALUE) {
            throw DatasetException("batchSize × contextLength exceeds the JVM array limit")
        }
    }

    fun readBatch(): TokenBatch? {
        val contextLength = windowReader.contextLength
        val inputs = IntArray(requestedBatchSize * contextLength)
        val targets = IntArray(requestedBatchSize * contextLength)
        val window = IntArray(contextLength + 1)
        var rows = 0

        while (rows < requestedBatchSize && windowReader.readWindow(window)) {
            val destinationOffset = rows * contextLength
            window.copyInto(inputs, destinationOffset, startIndex = 0, endIndex = contextLength)
            window.copyInto(targets, destinationOffset, startIndex = 1, endIndex = contextLength + 1)
            rows += 1
        }

        if (rows == 0 || (dropLastBatch && rows < requestedBatchSize)) return null
        return TokenBatch(
            batchSize = rows,
            sequenceLength = contextLength,
            inputIds = inputs.copyOf(rows * contextLength),
            targetIds = targets.copyOf(rows * contextLength),
        )
    }

    override fun close() = windowReader.close()
}
