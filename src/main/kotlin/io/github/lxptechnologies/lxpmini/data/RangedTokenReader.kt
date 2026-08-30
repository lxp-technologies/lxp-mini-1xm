package io.github.lxptechnologies.lxpmini.data

class RangedTokenReader(
    private val delegate: TokenReader,
    private val range: TokenRange,
) : TokenReader {
    private val skipBuffer = IntArray(SKIP_BUFFER_SIZE)
    private var initialized = false
    private var remaining = range.size

    override fun read(destination: IntArray, offset: Int, length: Int): Int {
        if (offset < 0 || length < 0 || offset > destination.size - length) {
            throw DatasetException("Invalid destination range: offset=$offset, length=$length, size=${destination.size}")
        }
        if (!initialized) skipPrefix()
        if (remaining == 0L || length == 0) return 0

        val requested = minOf(length.toLong(), remaining).toInt()
        val count = delegate.read(destination, offset, requested)
        if (count == 0) {
            throw DatasetException("Corpus ended before token ${range.endExclusive}")
        }
        remaining -= count
        return count
    }

    override fun close() = delegate.close()

    private fun skipPrefix() {
        var toSkip = range.startInclusive
        while (toSkip > 0) {
            val count = delegate.read(skipBuffer, length = minOf(toSkip, skipBuffer.size.toLong()).toInt())
            if (count == 0) throw DatasetException("Corpus ended before token ${range.startInclusive}")
            toSkip -= count
        }
        initialized = true
    }

    private companion object {
        const val SKIP_BUFFER_SIZE = 8192
    }
}
