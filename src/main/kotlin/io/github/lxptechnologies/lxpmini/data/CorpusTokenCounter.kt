package io.github.lxptechnologies.lxpmini.data

fun interface TokenReaderFactory {
    fun open(): TokenReader
}

object CorpusTokenCounter {
    fun count(readerFactory: TokenReaderFactory, bufferSize: Int = DEFAULT_BUFFER_SIZE): Long {
        if (bufferSize <= 0) throw DatasetException("bufferSize must be positive")
        val buffer = IntArray(bufferSize)
        var total = 0L
        readerFactory.open().use { reader ->
            while (true) {
                val count = reader.read(buffer)
                if (count == 0) break
                total += count
            }
        }
        return total
    }

    private const val DEFAULT_BUFFER_SIZE = 8192
}
