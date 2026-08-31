package io.github.lxptechnologies.lxpmini.generation

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.ln

class TokenSamplerTest {
    @Test
    fun `greedy chooses the largest logit and breaks ties by token ID`() {
        val sampler = TokenSampler(42)

        val result = sampler.select(
            floatArrayOf(1f, 3f, 3f, 2f),
            SamplingOptions(strategy = SamplingStrategy.GREEDY),
        )

        assertThat(result.tokenId).isEqualTo(1)
        assertThat(result.probability).isEqualTo(1.0)
        assertThat(result.candidates).containsExactly(TokenProbability(1, 1.0, 3.0))
    }

    @Test
    fun `temperature changes a calculable two-token distribution`() {
        val logits = floatArrayOf(0f, ln(4.0).toFloat())

        val cold = TokenSampler(1).select(logits, SamplingOptions(temperature = 1.0))
        val warm = TokenSampler(1).select(logits, SamplingOptions(temperature = 2.0))

        assertThat(cold.candidates.single { it.tokenId == 1 }.probability)
            .isCloseTo(0.8, offset(1e-7))
        assertThat(warm.candidates.single { it.tokenId == 1 }.probability)
            .isCloseTo(2.0 / 3.0, offset(1e-7))
        assertThat(warm.candidates.single { it.tokenId == 1 }.scaledLogit)
            .isCloseTo(ln(2.0), offset(1e-7))
    }

    @Test
    fun `top k keeps only k candidates and renormalizes`() {
        val result = TokenSampler(7).select(
            floatArrayOf(4f, 3f, 2f, 1f),
            SamplingOptions(topK = 2),
        )

        assertThat(result.candidates.map(TokenProbability::tokenId)).containsExactly(0, 1)
        assertThat(result.candidates.sumOf(TokenProbability::probability)).isCloseTo(1.0, offset(1e-12))
    }

    @Test
    fun `top p keeps the smallest prefix reaching the threshold and renormalizes`() {
        val result = TokenSampler(9).select(
            floatArrayOf(ln(0.60).toFloat(), ln(0.25).toFloat(), ln(0.15).toFloat()),
            SamplingOptions(topP = 0.70),
        )

        assertThat(result.candidates.map(TokenProbability::tokenId)).containsExactly(0, 1)
        assertThat(result.candidates.sumOf(TokenProbability::probability)).isCloseTo(1.0, offset(1e-12))
        assertThat(result.candidates.first().probability).isCloseTo(0.60 / 0.85, offset(1e-7))
    }

    @Test
    fun `same seed produces the same categorical sequence`() {
        val first = TokenSampler(123)
        val second = TokenSampler(123)
        val logits = floatArrayOf(0f, 0f, 0f)
        val options = SamplingOptions()

        val firstSequence = List(20) { first.select(logits, options).tokenId }
        val secondSequence = List(20) { second.select(logits, options).tokenId }

        assertThat(firstSequence).isEqualTo(secondSequence)
        assertThat(firstSequence.distinct()).hasSizeGreaterThan(1)
    }

    @Test
    fun `selection considers only tokens allowed by the constraint`() {
        val allowed = booleanArrayOf(true, false, true, false)
        val logits = floatArrayOf(1f, 100f, 2f, 200f)

        val greedy = TokenSampler(1).select(
            logits,
            SamplingOptions(strategy = SamplingStrategy.GREEDY),
            allowed,
        )
        val sampled = TokenSampler(1).select(logits, SamplingOptions(), allowed)

        assertThat(greedy.tokenId).isEqualTo(2)
        assertThat(sampled.candidates.map(TokenProbability::tokenId)).containsOnly(0, 2)
    }

    @Test
    fun `rejects invalid sampling controls and non-finite logits`() {
        assertThatThrownBy { TokenSampler(1).select(floatArrayOf(Float.NaN), SamplingOptions()) }
            .isInstanceOf(GenerationException::class.java)
        assertThatThrownBy { TokenSampler(1).select(floatArrayOf(1f), SamplingOptions(temperature = 0.0)) }
            .isInstanceOf(GenerationException::class.java)
        assertThatThrownBy { TokenSampler(1).select(floatArrayOf(1f), SamplingOptions(topK = 2)) }
            .isInstanceOf(GenerationException::class.java)
        assertThatThrownBy { TokenSampler(1).select(floatArrayOf(1f), SamplingOptions(topP = 0.0)) }
            .isInstanceOf(GenerationException::class.java)
        assertThatThrownBy {
            TokenSampler(1).select(floatArrayOf(1f), SamplingOptions(), booleanArrayOf(false))
        }.isInstanceOf(GenerationException::class.java)
    }
}
