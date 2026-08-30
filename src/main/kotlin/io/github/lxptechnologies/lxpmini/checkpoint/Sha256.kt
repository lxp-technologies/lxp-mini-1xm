package io.github.lxptechnologies.lxpmini.checkpoint

import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest

object Sha256 {
    fun of(path: Path): String = try {
        Files.newInputStream(path).use { input ->
            val digest = MessageDigest.getInstance(ALGORITHM)
            val buffer = ByteArray(BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
            digest.digest().hex()
        }
    } catch (exception: IOException) {
        throw CheckpointException("Cannot checksum $path: ${exception.message}", exception)
    }

    fun of(bytes: ByteArray): String = MessageDigest.getInstance(ALGORITHM).digest(bytes).hex()

    fun isValid(value: String): Boolean = value.matches(SHA_256_PATTERN)

    private fun ByteArray.hex(): String = joinToString("") { byte -> "%02x".format(byte.toInt() and 0xff) }

    private const val ALGORITHM = "SHA-256"
    private const val BUFFER_SIZE = 8192
    private val SHA_256_PATTERN = Regex("[0-9a-f]{64}")
}
