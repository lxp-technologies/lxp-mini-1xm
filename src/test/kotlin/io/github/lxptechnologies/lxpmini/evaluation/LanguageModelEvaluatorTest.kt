package io.github.lxptechnologies.lxpmini.evaluation

import ai.djl.Device
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import io.github.lxptechnologies.lxpmini.data.TokenBatch
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.ln

class LanguageModelEvaluatorTest {
    @Test
    fun `perplexity is the exponential of mean cross entropy`() {
        assertThat(LanguageModelEvaluator.perplexity(ln(4.0))).isCloseTo(4.0, offset(1e-12))
        assertThatThrownBy { LanguageModelEvaluator.perplexity(-0.1) }
            .isInstanceOf(EvaluationException::class.java)
    }

    @Test
    fun `evaluation is token weighted measures throughput and creates no gradients`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, tinyConfig()).use { model ->
                model.initialize(manager, DataType.FLOAT32, Shape(2, 3))
                val times = ArrayDeque(listOf(2_000_000_000L, 3_000_000_000L))
                val evaluator = LanguageModelEvaluator(model, manager, nanoTime = times::removeFirst)
                val parametersBefore = model.parameters.values().associate { parameter ->
                    parameter.id to parameter.array.toFloatArray()
                }
                val gradientsBefore = model.parameters.values().associate { parameter ->
                    parameter.id to parameter.array.takeIf { it.hasGradient() }?.gradient?.toFloatArray()
                }
                val batches = sequenceOf(
                    TokenBatch(2, 3, intArrayOf(3, 4, 5, 4, 5, 6), intArrayOf(4, 5, 6, 5, 6, 7)),
                    TokenBatch(1, 3, intArrayOf(3, 4, 5), intArrayOf(4, 5, 6)),
                )

                val result = evaluator.evaluate(batches)

                assertThat(result.batchCount).isEqualTo(2)
                assertThat(result.tokenCount).isEqualTo(9)
                assertThat(result.elapsedSeconds).isEqualTo(1.0)
                assertThat(result.tokensPerSecond).isEqualTo(9.0)
                assertThat(result.loss).isFinite().isPositive()
                assertThat(result.perplexity).isCloseTo(kotlin.math.exp(result.loss), offset(1e-12))
                model.parameters.values().forEach { parameter ->
                    assertThat(parameter.array.toFloatArray()).containsExactly(*parametersBefore.getValue(parameter.id))
                    val gradientAfter = parameter.array.takeIf { it.hasGradient() }?.gradient?.toFloatArray()
                    assertThat(gradientAfter).isEqualTo(gradientsBefore.getValue(parameter.id))
                }
            }
        }
    }

    private fun tinyConfig() = ModelConfig(
        vocabSize = 16,
        contextLength = 3,
        dModel = 8,
        numLayers = 1,
        numHeads = 2,
        ffnDim = 16,
        ropeTheta = 10_000.0,
        dropout = 0.0,
        tieEmbeddings = true,
    )
}
