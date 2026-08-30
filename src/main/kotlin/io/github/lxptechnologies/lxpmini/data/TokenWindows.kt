package io.github.lxptechnologies.lxpmini.data

import java.util.Random

interface WindowReader : AutoCloseable {
    val contextLength: Int

    fun readWindow(destination: IntArray): Boolean
}

data class WindowPlan(
    val tokenCount: Long,
    val contextLength: Int,
    val stride: Int,
) {
    init {
        if (tokenCount < 0) throw DatasetException("tokenCount must be non-negative")
        if (contextLength <= 0 || contextLength == Int.MAX_VALUE) {
            throw DatasetException("contextLength must be between 1 and ${Int.MAX_VALUE - 1}")
        }
        if (stride <= 0) throw DatasetException("stride must be positive")
    }

    val windowSize: Int
        get() = contextLength + 1

    val windowCount: Long
        get() = if (tokenCount < windowSize) 0 else 1 + (tokenCount - windowSize) / stride

    val trailingTokenCount: Long
        get() = if (windowCount == 0L) tokenCount else {
            tokenCount - ((windowCount - 1) * stride + windowSize)
        }
}

class SlidingWindowReader(
    private val tokenReader: TokenReader,
    override val contextLength: Int,
    private val stride: Int = contextLength,
) : WindowReader {
    private val window = IntArray(requirePositiveContextLength(contextLength) + 1)
    private var firstWindow = true
    private var exhausted = false

    init {
        if (contextLength <= 0) throw DatasetException("contextLength must be positive")
        if (stride <= 0) throw DatasetException("stride must be positive")
    }

    override fun readWindow(destination: IntArray): Boolean {
        if (destination.size < window.size) {
            throw DatasetException("Window destination must contain at least ${window.size} tokens")
        }
        if (exhausted) return false

        val complete = if (firstWindow) {
            firstWindow = false
            readExactly(window, 0, window.size)
        } else {
            advanceWindow()
        }
        if (!complete) {
            exhausted = true
            return false
        }
        window.copyInto(destination, endIndex = window.size)
        return true
    }

    override fun close() = tokenReader.close()

    private fun advanceWindow(): Boolean {
        if (stride < window.size) {
            window.copyInto(window, startIndex = stride, endIndex = window.size)
            return readExactly(window, window.size - stride, stride)
        }

        var toSkip = stride - window.size
        val skipBuffer = IntArray(minOf(toSkip, SKIP_BUFFER_SIZE))
        while (toSkip > 0) {
            val count = tokenReader.read(skipBuffer, length = minOf(toSkip, skipBuffer.size))
            if (count == 0) return false
            toSkip -= count
        }
        return readExactly(window, 0, window.size)
    }

    private fun readExactly(destination: IntArray, offset: Int, length: Int): Boolean {
        var count = 0
        while (count < length) {
            val read = tokenReader.read(destination, offset + count, length - count)
            if (read == 0) return false
            count += read
        }
        return true
    }

    private companion object {
        const val SKIP_BUFFER_SIZE = 8192

        fun requirePositiveContextLength(value: Int): Int {
            if (value <= 0 || value == Int.MAX_VALUE) {
                throw DatasetException("contextLength must be between 1 and ${Int.MAX_VALUE - 1}")
            }
            return value
        }
    }
}

class BufferedShufflingWindowReader(
    private val delegate: WindowReader,
    bufferSize: Int,
    seed: Long,
) : WindowReader {
    override val contextLength: Int = delegate.contextLength
    private val requestedBufferSize = requirePositiveBufferSize(bufferSize)
    private val random = Random(seed)
    private val buffer = ArrayList<IntArray>(requestedBufferSize)
    private var delegateExhausted = false
    private var initialized = false

    override fun readWindow(destination: IntArray): Boolean {
        if (destination.size < contextLength + 1) {
            throw DatasetException("Window destination must contain at least ${contextLength + 1} tokens")
        }
        if (!initialized) initializeBuffer()
        if (buffer.isEmpty()) return false

        val selectedIndex = random.nextInt(buffer.size)
        buffer[selectedIndex].copyInto(destination)

        val replacement = IntArray(contextLength + 1)
        if (!delegateExhausted && delegate.readWindow(replacement)) {
            buffer[selectedIndex] = replacement
        } else {
            delegateExhausted = true
            val last = buffer.removeAt(buffer.lastIndex)
            if (selectedIndex < buffer.size) buffer[selectedIndex] = last
        }
        return true
    }

    override fun close() = delegate.close()

    private fun initializeBuffer() {
        initialized = true
        while (buffer.size < requestedBufferSize) {
            val window = IntArray(contextLength + 1)
            if (!delegate.readWindow(window)) {
                delegateExhausted = true
                break
            }
            buffer += window
        }
    }

    private companion object {
        fun requirePositiveBufferSize(value: Int): Int {
            if (value <= 0) throw DatasetException("shuffle bufferSize must be positive")
            return value
        }
    }
}
