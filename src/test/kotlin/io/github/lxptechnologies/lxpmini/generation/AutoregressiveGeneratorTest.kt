package io.github.lxptechnologies.lxpmini.generation

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class AutoregressiveGeneratorTest {
    @Test
    fun `stops immediately after sampling EOS`() {
        var calls = 0
        val generator = AutoregressiveGenerator(4, 5, TokenSampler(1)) {
            calls += 1
            if (calls == 1) floatArrayOf(0f, 5f, 0f, 0f, 0f) else floatArrayOf(0f, 0f, 5f, 0f, 0f)
        }

        val result = generator.generate(
            intArrayOf(3),
            maxNewTokens = 10,
            eosTokenId = 2,
            options = SamplingOptions(strategy = SamplingStrategy.GREEDY),
        )

        assertThat(result.generatedTokenIds).containsExactly(1, 2)
        assertThat(result.stoppedByEos).isTrue()
        assertThat(result.steps).hasSize(2)
        assertThat(calls).isEqualTo(2)
    }

    @Test
    fun `slides to the last context tokens while retaining the complete result`() {
        val observedContexts = mutableListOf<IntArray>()
        val generator = AutoregressiveGenerator(2, 6, TokenSampler(1)) { context ->
            observedContexts += context.copyOf()
            floatArrayOf(0f, 5f, 0f, 0f, 0f, 0f)
        }

        val result = generator.generate(
            intArrayOf(3, 4, 5),
            maxNewTokens = 3,
            eosTokenId = 2,
            options = SamplingOptions(strategy = SamplingStrategy.GREEDY),
        )

        assertThat(observedContexts.map(IntArray::toList)).containsExactly(
            listOf(4, 5),
            listOf(5, 1),
            listOf(1, 1),
        )
        assertThat(result.allTokenIds).containsExactly(3, 4, 5, 1, 1, 1)
        assertThat(result.stoppedByEos).isFalse()
    }
}
