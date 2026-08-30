package io.github.lxptechnologies.lxpmini.data

interface TokenReader : AutoCloseable {
    fun read(
        destination: IntArray,
        offset: Int = 0,
        length: Int = destination.size - offset,
    ): Int
}

class DatasetException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
