package io.github.lxptechnologies.lxpmini.inference

import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class Utf8TokenConstraintTest {
    private val tokenizer = ByteTokenizer()
    private val constraint = Utf8TokenConstraint(
        tokenizer,
        tokenizer.vocabularySize,
        SpecialToken.EOS.id,
    )

    @Test
    fun `allows ASCII and valid leading bytes but rejects malformed starts`() {
        val allowed = constraint.allowedTokenIds(intArrayOf(), remainingSteps = 3)

        assertThat(allowed[token('A'.code)]).isTrue()
        assertThat(allowed[token(0xC3)]).isTrue()
        assertThat(allowed[token(0xF0)]).isTrue()
        assertThat(allowed[token(0x80)]).isFalse()
        assertThat(allowed[token(0xC0)]).isFalse()
        assertThat(allowed[token(0xFF)]).isFalse()
    }

    @Test
    fun `allows only a continuation after a leading byte`() {
        val prefix = intArrayOf(token(0xC3))
        val allowed = constraint.allowedTokenIds(prefix, remainingSteps = 0)

        assertThat(allowed[token(0xA9)]).isTrue()
        assertThat(allowed[token('A'.code)]).isFalse()
        assertThat(allowed[SpecialToken.EOS.id]).isFalse()
    }

    @Test
    fun `does not begin a character that cannot finish inside the token budget`() {
        val finalStep = constraint.allowedTokenIds(intArrayOf(), remainingSteps = 0)
        val twoStepsLeft = constraint.allowedTokenIds(intArrayOf(), remainingSteps = 1)

        assertThat(finalStep[token(0xC3)]).isFalse()
        assertThat(finalStep[token('A'.code)]).isTrue()
        assertThat(twoStepsLeft[token(0xC3)]).isTrue()
        assertThat(twoStepsLeft[token(0xE0)]).isFalse()
    }

    @Test
    fun `enforces Unicode scalar boundaries`() {
        val afterE0 = constraint.allowedTokenIds(intArrayOf(token(0xE0)), remainingSteps = 1)
        val afterEd = constraint.allowedTokenIds(intArrayOf(token(0xED)), remainingSteps = 1)
        val afterF4 = constraint.allowedTokenIds(intArrayOf(token(0xF4)), remainingSteps = 2)

        assertThat(afterE0[token(0x9F)]).isFalse()
        assertThat(afterE0[token(0xA0)]).isTrue()
        assertThat(afterEd[token(0x9F)]).isTrue()
        assertThat(afterEd[token(0xA0)]).isFalse()
        assertThat(afterF4[token(0x8F)]).isTrue()
        assertThat(afterF4[token(0x90)]).isFalse()
    }

    private fun token(byte: Int): Int = tokenizer.tokenIdForByte(byte)
}
