package io.github.lxptechnologies.lxpmini.data

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizer
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path

class StreamingBpeTokenReader private constructor(
    private val input: InputStream,
    private val tokenizer: BpeTokenizer,
    byteChunkSize: Int,
) : TokenReader {
    private val byteBuffer = ByteArray(requirePositiveChunkSize(byteChunkSize))
    private var carry = byteArrayOf()
    private var pendingTokens = intArrayOf()
    private var pendingOffset = 0
    private var inputExhausted = false
    private var finalCarryFlushed = false
    private var closed = false

    constructor(
        path: Path,
        tokenizer: BpeTokenizer,
        byteChunkSize: Int = DEFAULT_BYTE_CHUNK_SIZE,
    ) : this(open(path), tokenizer, byteChunkSize)

    override fun read(destination: IntArray, offset: Int, length: Int): Int {
        requireOpen()
        requireValidRange(destination, offset, length)
        if (length == 0) return 0

        var written = 0
        while (written < length) {
            if (pendingOffset >= pendingTokens.size && !refillPendingTokens()) break

            val count = minOf(length - written, pendingTokens.size - pendingOffset)
            pendingTokens.copyInto(
                destination = destination,
                destinationOffset = offset + written,
                startIndex = pendingOffset,
                endIndex = pendingOffset + count,
            )
            pendingOffset += count
            written += count
        }
        return written
    }

    override fun close() {
        if (!closed) {
            closed = true
            try {
                input.close()
            } catch (exception: IOException) {
                throw DatasetException("Cannot close corpus stream: ${exception.message}", exception)
            }
        }
    }

    private fun refillPendingTokens(): Boolean {
        pendingTokens = intArrayOf()
        pendingOffset = 0

        while (pendingTokens.isEmpty()) {
            if (inputExhausted) return flushFinalCarry()

            val bytesRead = try {
                input.read(byteBuffer)
            } catch (exception: IOException) {
                throw DatasetException("Cannot read corpus stream: ${exception.message}", exception)
            }
            if (bytesRead < 0) {
                inputExhausted = true
                continue
            }
            if (bytesRead == 0) continue

            val combined = ByteArray(carry.size + bytesRead)
            carry.copyInto(combined)
            byteBuffer.copyInto(combined, destinationOffset = carry.size, endIndex = bytesRead)
            splitSafePrefix(combined)
        }
        return true
    }

    private fun splitSafePrefix(bytes: ByteArray) {
        val encoded = tokenizer.encodeBytes(bytes)
        val safeByteLimit = (bytes.size - tokenizer.maximumTokenByteLength).coerceAtLeast(0)
        var emittedBytes = 0
        var emittedTokens = 0

        while (emittedTokens < encoded.size) {
            val nextEnd = emittedBytes + tokenizer.bytesForToken(encoded[emittedTokens]).size
            if (nextEnd > safeByteLimit) break
            emittedBytes = nextEnd
            emittedTokens += 1
        }

        pendingTokens = encoded.copyOf(emittedTokens)
        carry = bytes.copyOfRange(emittedBytes, bytes.size)
    }

    private fun flushFinalCarry(): Boolean {
        if (finalCarryFlushed) return false
        finalCarryFlushed = true
        pendingTokens = tokenizer.encodeBytes(carry)
        pendingOffset = 0
        carry = byteArrayOf()
        return pendingTokens.isNotEmpty()
    }

    private fun requireOpen() {
        if (closed) throw DatasetException("Token reader is closed")
    }

    private fun requireValidRange(destination: IntArray, offset: Int, length: Int) {
        if (offset < 0 || length < 0 || offset > destination.size - length) {
            throw DatasetException(
                "Invalid destination range: offset=$offset, length=$length, size=${destination.size}",
            )
        }
    }

    companion object {
        const val DEFAULT_BYTE_CHUNK_SIZE = 64 * 1024

        private fun open(path: Path): InputStream = try {
            Files.newInputStream(path)
        } catch (exception: IOException) {
            throw DatasetException("Cannot open corpus $path: ${exception.message}", exception)
        }

        private fun requirePositiveChunkSize(value: Int): Int {
            if (value <= 0) throw DatasetException("byteChunkSize must be positive")
            return value
        }
    }
}
