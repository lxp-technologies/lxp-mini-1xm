package io.github.lxptechnologies.lxpmini.tokenizer

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.ValueSource

class ByteTokenizerTest {
    private val tokenizer = ByteTokenizer()

    @Test
    fun `maps ASCII bytes to offset token IDs`() {
        val tokenIds = tokenizer.encode("ABC")

        assertThat(tokenIds).containsExactly(68, 69, 70)
        assertThat(tokenizer.decode(tokenIds)).isEqualTo("ABC")
    }

    @Test
    fun `exposes the UTF-8 bytes behind accents and emoji`() {
        assertThat(tokenizer.encode("é")).containsExactly(198, 172)
        assertThat(tokenizer.encode("👋")).containsExactly(243, 162, 148, 142)
    }

    @ParameterizedTest(name = "round-trip: {0}")
    @ValueSource(
        strings = [
            "",
            "plain ASCII",
            "Bonjour, ça va?",
            "Québec\nMontréal",
            "👋🌍",
            "漢字とかな",
            "é",
            " spaces\tand\nnewlines ",
            "\u0000",
        ],
    )
    fun `round-trips valid Unicode text`(text: String) {
        assertThat(tokenizer.decode(tokenizer.encode(text))).isEqualTo(text)
    }

    @Test
    fun `adds explicit BOS and EOS tokens only when requested`() {
        val tokenIds = tokenizer.encode("A", addBos = true, addEos = true)

        assertThat(tokenIds).containsExactly(SpecialToken.BOS.id, 68, SpecialToken.EOS.id)
        assertThat(tokenizer.decode(tokenIds)).isEqualTo("A")
    }

    @Test
    fun `skips padding and sequence markers during normal decoding`() {
        val tokenIds = intArrayOf(
            SpecialToken.PAD.id,
            SpecialToken.BOS.id,
            tokenizer.tokenIdForByte('A'.code),
            SpecialToken.EOS.id,
            SpecialToken.PAD.id,
        )

        assertThat(tokenizer.decode(tokenIds)).isEqualTo("A")
    }

    @Test
    fun `represents every possible byte without an unknown token`() {
        val allByteValues = ByteArray(256) { it.toByte() }
        val allByteTokenIds = IntArray(256) { tokenizer.tokenIdForByte(it) }

        assertThat(allByteTokenIds).containsExactly(*(3..258).toList().toIntArray())
        assertThat(tokenizer.decodeToBytes(allByteTokenIds)).containsExactly(*allByteValues)
        assertThat(tokenizer.vocabularySize).isEqualTo(259)
    }

    @Test
    fun `rejects token IDs outside the vocabulary`() {
        assertThatThrownBy { tokenizer.decode(intArrayOf(259)) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("outside the byte tokenizer vocabulary")
    }

    @Test
    fun `rejects bytes outside the unsigned byte range`() {
        assertThatThrownBy { tokenizer.tokenIdForByte(256) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessage("Byte value must be in 0..255, got 256")
    }

    @Test
    fun `rejects byte token sequences that are not valid UTF-8 text`() {
        val incompleteTwoByteSequence = tokenizer.tokenIdForByte(0xC3)

        assertThatThrownBy { tokenizer.decode(intArrayOf(incompleteTwoByteSequence)) }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessage("Token IDs do not form valid UTF-8")
    }

    @Test
    fun `requires explicit skipping when special tokens are present`() {
        assertThatThrownBy {
            tokenizer.decodeToBytes(intArrayOf(SpecialToken.BOS.id), skipSpecialTokens = false)
        }
            .isInstanceOf(TokenizerException::class.java)
            .hasMessageContaining("Cannot decode special token <bos>")
    }
}
