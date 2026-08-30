package io.github.lxptechnologies.lxpmini.data

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerTrainer
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import kotlin.io.path.writeBytes

class StreamingBpeTokenReaderTest {
    @Test
    fun `matches full encoding across every byte chunk boundary`(@TempDir directory: Path) {
        val text = "abab Bonjour bonjour, été! 👋 abab"
        val bytes = text.toByteArray(StandardCharsets.UTF_8)
        val tokenizer = BpeTokenizerTrainer().train(bytes, targetVocabularySize = 275).tokenizer
        val corpusPath = directory.resolve("corpus.txt")
        corpusPath.writeBytes(bytes)
        val expected = tokenizer.encodeBytes(bytes)

        for (chunkSize in 1..bytes.size + 1) {
            val actual = StreamingBpeTokenReader(corpusPath, tokenizer, chunkSize).use(::readAll)
            assertThat(actual).describedAs("byte chunk size $chunkSize").isEqualTo(expected)
        }
    }

    @Test
    fun `supports partial destination writes and reaches end of stream`(@TempDir directory: Path) {
        val bytes = "abababab".toByteArray(StandardCharsets.UTF_8)
        val tokenizer = BpeTokenizerTrainer().train(bytes, targetVocabularySize = 261).tokenizer
        val corpusPath = directory.resolve("corpus.txt")
        corpusPath.writeBytes(bytes)
        val destination = IntArray(4) { -1 }

        StreamingBpeTokenReader(corpusPath, tokenizer, byteChunkSize = 2).use { reader ->
            assertThat(reader.read(destination, offset = 1, length = 2)).isEqualTo(2)
            assertThat(destination).containsExactly(-1, 260, 260, -1)
            assertThat(reader.read(destination)).isZero()
        }
    }

    @Test
    fun `rejects reads after close`(@TempDir directory: Path) {
        val bytes = "abc".toByteArray(StandardCharsets.UTF_8)
        val tokenizer = BpeTokenizerTrainer().train(bytes, targetVocabularySize = 259).tokenizer
        val reader = StreamingBpeTokenReader(directory.resolve("corpus.txt").also { it.writeBytes(bytes) }, tokenizer)
        reader.close()

        assertThatThrownBy { reader.read(IntArray(1)) }
            .isInstanceOf(DatasetException::class.java)
            .hasMessage("Token reader is closed")
    }

    private fun readAll(reader: TokenReader): IntArray {
        val chunks = mutableListOf<IntArray>()
        val buffer = IntArray(7)
        var total = 0
        while (true) {
            val count = reader.read(buffer)
            if (count == 0) break
            chunks += buffer.copyOf(count)
            total += count
        }
        return IntArray(total).also { result ->
            var offset = 0
            chunks.forEach { chunk ->
                chunk.copyInto(result, destinationOffset = offset)
                offset += chunk.size
            }
        }
    }
}
