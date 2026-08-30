package io.github.lxptechnologies.lxpmini.training

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test

class WarmupCosineSchedulerTest {
    @Test
    fun `warms linearly then decays to the minimum on the final update`() {
        val scheduler = WarmupCosineScheduler(0.01f, 0.001f, warmupUpdates = 2, totalUpdates = 6)

        assertThat(scheduler.learningRateForUpdate(1)).isCloseTo(0.005f, offset(1e-7f))
        assertThat(scheduler.learningRateForUpdate(2)).isCloseTo(0.01f, offset(1e-7f))
        assertThat(scheduler.learningRateForUpdate(3)).isCloseTo(0.01f, offset(1e-7f))
        assertThat(scheduler.learningRateForUpdate(6)).isCloseTo(0.001f, offset(1e-7f))
        assertThat(scheduler.learningRateForUpdate(4)).isBetween(0.001f, 0.01f)
    }

    @Test
    fun `tracker uses DJL one based optimizer counts and clamps inference calls`() {
        val scheduler = WarmupCosineScheduler(0.01f, 0.001f, warmupUpdates = 2, totalUpdates = 6)

        assertThat(scheduler.getNewValue(1)).isEqualTo(scheduler.learningRateForUpdate(1))
        assertThat(scheduler.getNewValue(6)).isEqualTo(scheduler.learningRateForUpdate(6))
        assertThat(scheduler.getNewValue(0)).isEqualTo(scheduler.learningRateForUpdate(1))
        assertThat(scheduler.getNewValue(100)).isEqualTo(scheduler.learningRateForUpdate(6))
    }

    @Test
    fun `rejects invalid schedule boundaries`() {
        assertThatThrownBy { WarmupCosineScheduler(0.01f, 0.02f, 0, 10) }
            .isInstanceOf(TrainingException::class.java)
        assertThatThrownBy { WarmupCosineScheduler(0.01f, 0.001f, 10, 10) }
            .isInstanceOf(TrainingException::class.java)

        val scheduler = WarmupCosineScheduler(0.01f, 0.001f, 0, 10)
        assertThatThrownBy { scheduler.learningRateForUpdate(0) }
            .isInstanceOf(TrainingException::class.java)
    }
}
