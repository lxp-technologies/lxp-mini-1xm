package io.github.lxptechnologies.lxpmini.data

class IntArrayTokenReader(
    tokens: IntArray,
) : TokenReader {
    private val tokens = tokens.copyOf()
    private var tokenOffset = 0
    private var closed = false

    override fun read(destination: IntArray, offset: Int, length: Int): Int {
        if (closed) throw DatasetException("Token reader is closed")
        if (offset < 0 || length < 0 || offset > destination.size - length) {
            throw DatasetException("Invalid destination range: offset=$offset, length=$length, size=${destination.size}")
        }
        val count = minOf(length, tokens.size - tokenOffset)
        tokens.copyInto(
            destination,
            destinationOffset = offset,
            startIndex = tokenOffset,
            endIndex = tokenOffset + count,
        )
        tokenOffset += count
        return count
    }

    override fun close() {
        closed = true
    }
}
