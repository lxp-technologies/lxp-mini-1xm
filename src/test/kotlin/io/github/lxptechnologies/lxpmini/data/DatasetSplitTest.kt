package io.github.lxptechnologies.lxpmini.data

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class DatasetSplitTest {
    @Test
    fun `creates contiguous train and validation ranges without overlap`() {
        val split = DeterministicTokenSplit.contiguous(totalTokenCount = 10, validationFraction = 0.2)

        assertThat(split.train).isEqualTo(TokenRange(0, 8))
        assertThat(split.validation).isEqualTo(TokenRange(8, 10))
        assertThat(split.train.endExclusive).isEqualTo(split.validation.startInclusive)
    }

    @Test
    fun `rounds validation down and assigns every token exactly once`() {
        val split = DeterministicTokenSplit.contiguous(totalTokenCount = 11, validationFraction = 0.25)

        assertThat(split.train.size).isEqualTo(9)
        assertThat(split.validation.size).isEqualTo(2)
        assertThat(split.train.size + split.validation.size).isEqualTo(11)
    }

    @Test
    fun `rejects invalid validation fractions`() {
        assertThatThrownBy { DeterministicTokenSplit.contiguous(10, 1.1) }
            .isInstanceOf(DatasetException::class.java)
            .hasMessageContaining("between 0.0 and 1.0")
    }
}
