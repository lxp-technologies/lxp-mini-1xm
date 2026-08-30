package io.github.lxptechnologies.lxpmini.training

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import kotlin.math.ln

class NextTokenCrossEntropyTest {
    private val loss = NextTokenCrossEntropy()

    @Test
    fun `uniform logits have loss log vocabulary size`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val logits = manager.zeros(Shape(1, 2, 4))
            val targets = manager.create(longArrayOf(0, 3), Shape(1, 2))

            val value = loss.evaluate(targets, logits).getFloat()

            assertThat(value).isCloseTo(ln(4.0).toFloat(), offset(1e-6f))
        }
    }

    @Test
    fun `known binary logits match manual negative log likelihood and backpropagate`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val logits = manager.create(floatArrayOf(2f, 0f), Shape(1, 1, 2))
            val targets = manager.create(longArrayOf(0), Shape(1, 1))
            logits.setRequiresGradient(true)

            val value = Engine.getInstance().newGradientCollector().use { collector ->
                loss.evaluate(targets, logits).also { collector.backward(it) }.getFloat()
            }

            assertThat(value).isCloseTo(ln(1.0 + kotlin.math.exp(-2.0)).toFloat(), offset(1e-6f))
            assertThat(logits.gradient.toFloatArray().all(Float::isFinite)).isTrue()
        }
    }

    @Test
    fun `rejects target shapes that do not match logits`() {
        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val logits = manager.zeros(Shape(2, 3, 4))
            val targets = manager.zeros(Shape(2, 2))

            assertThatThrownBy { loss.evaluate(targets, logits) }
                .isInstanceOf(TrainingException::class.java)
                .hasMessageContaining("must match logits")
        }
    }
}
