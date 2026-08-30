package io.github.lxptechnologies.lxpmini.tokenizer

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import java.nio.charset.StandardCharsets

class BpeTokenizerTrainerTest {
    private val trainer = BpeTokenizerTrainer()

    @Test
    fun `learns the most frequent pairs and assigns sequential token IDs`() {
        val trained = trainer.train("abab".utf8Bytes(), targetVocabularySize = 261)

        assertThat(trained.tokenizer.merges).containsExactly(
            BpeMerge(TokenPair(100, 101), resultId = 259, trainingFrequency = 2),
            BpeMerge(TokenPair(259, 259), resultId = 260, trainingFrequency = 1),
        )
        assertThat(trained.tokenizer.bytesForToken(259)).isEqualTo("ab".utf8Bytes())
        assertThat(trained.tokenizer.bytesForToken(260)).isEqualTo("abab".utf8Bytes())
        assertThat(trained.tokenizer.encode("abab")).containsExactly(260)
    }

    @Test
    fun `breaks frequency ties using the lexicographically smallest pair`() {
        val trained = trainer.train("abac".utf8Bytes(), targetVocabularySize = 260)

        assertThat(trained.tokenizer.merges.single().pair).isEqualTo(TokenPair(100, 101))
    }

    @Test
    fun `replaces overlapping occurrences from left to right without overlap`() {
        val trained = trainer.train("aaa".utf8Bytes(), targetVocabularySize = 260)

        assertThat(trained.tokenizer.encode("aaa")).containsExactly(259, 100)
    }

    @Test
    fun `round trips Unicode after learning byte merges`() {
        val text = "Allo, monde! Bonjour, monde!"
        val trained = trainer.train(text.utf8Bytes(), targetVocabularySize = 270)

        assertThat(trained.tokenizer.decode(trained.tokenizer.encode(text))).isEqualTo(text)
        assertThat(trained.tokenizer.compressionRatio(text)).isGreaterThan(1.0)
    }

    @Test
    fun `produces deterministic merges and corpus metadata`() {
        val corpus = "deterministic deterministic".utf8Bytes()

        val first = trainer.train(corpus, targetVocabularySize = 266)
        val second = trainer.train(corpus, targetVocabularySize = 266)

        assertThat(first.metadata).isEqualTo(second.metadata)
        assertThat(first.tokenizer.merges).isEqualTo(second.tokenizer.merges)
        assertThat(first.tokenizer.encode("deterministic")).isEqualTo(second.tokenizer.encode("deterministic"))
        assertThat(first.metadata.corpusByteCount).isEqualTo(corpus.size.toLong())
        assertThat(first.metadata.corpusSha256).matches("[0-9a-f]{64}")
    }

    @Test
    fun `accepts the base vocabulary without requiring corpus pairs`() {
        val trained = trainer.train(byteArrayOf(), targetVocabularySize = ByteTokenizer.VOCABULARY_SIZE)

        assertThat(trained.tokenizer.vocabularySize).isEqualTo(259)
        assertThat(trained.tokenizer.merges).isEmpty()
    }

    @Test
    fun `rejects target sizes below the complete byte vocabulary`() {
        assertThatThrownBy { trainer.train("abc".utf8Bytes(), targetVocabularySize = 258) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("at least 259")
    }

    @Test
    fun `explains when the corpus cannot reach the requested vocabulary size`() {
        assertThatThrownBy { trainer.train("a".utf8Bytes(), targetVocabularySize = 260) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("training stopped at 259")
    }

    private fun String.utf8Bytes(): ByteArray = toByteArray(StandardCharsets.UTF_8)
}
