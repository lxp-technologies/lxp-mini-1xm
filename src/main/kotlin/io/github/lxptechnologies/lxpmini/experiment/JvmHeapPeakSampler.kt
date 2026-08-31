package io.github.lxptechnologies.lxpmini.experiment

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

class JvmHeapPeakSampler(
    samplePeriodMillis: Long = 5,
) : AutoCloseable {
    private val samplePeriodMillis = requirePositive(samplePeriodMillis)
    private val runtime = Runtime.getRuntime()
    private val running = AtomicBoolean(true)
    val baselineUsedBytes: Long = usedBytes()
    private val peak = AtomicLong(baselineUsedBytes)
    private val thread = Thread(::sampleUntilClosed, "lxp-mini-heap-sampler").apply {
        isDaemon = true
        start()
    }

    val peakUsedBytes: Long
        get() = peak.get()

    val peakDeltaBytes: Long
        get() = (peakUsedBytes - baselineUsedBytes).coerceAtLeast(0)

    override fun close() {
        sample()
        running.set(false)
        thread.join(samplePeriodMillis * 4)
        sample()
    }

    private fun sampleUntilClosed() {
        while (running.get()) {
            sample()
            try {
                Thread.sleep(samplePeriodMillis)
            } catch (_: InterruptedException) {
                Thread.currentThread().interrupt()
                return
            }
        }
    }

    private fun sample() {
        val current = usedBytes()
        peak.accumulateAndGet(current, ::maxOf)
    }

    private fun usedBytes(): Long = runtime.totalMemory() - runtime.freeMemory()

    private companion object {
        fun requirePositive(value: Long): Long {
            if (value <= 0) throw ScaleExperimentException("samplePeriodMillis must be positive")
            return value
        }
    }
}
