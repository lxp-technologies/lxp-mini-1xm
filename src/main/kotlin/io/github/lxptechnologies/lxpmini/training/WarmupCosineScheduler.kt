package io.github.lxptechnologies.lxpmini.training

import ai.djl.training.tracker.Tracker
import kotlin.math.PI
import kotlin.math.cos

class WarmupCosineScheduler(
    val maximumLearningRate: Float,
    val minimumLearningRate: Float,
    val warmupUpdates: Int,
    val totalUpdates: Int,
) : Tracker {
    init {
        if (!maximumLearningRate.isFinite() || maximumLearningRate <= 0f) {
            throw TrainingException("maximumLearningRate must be finite and positive")
        }
        if (!minimumLearningRate.isFinite() || minimumLearningRate < 0f) {
            throw TrainingException("minimumLearningRate must be finite and non-negative")
        }
        if (minimumLearningRate > maximumLearningRate) {
            throw TrainingException("minimumLearningRate cannot exceed maximumLearningRate")
        }
        if (totalUpdates <= 0) throw TrainingException("totalUpdates must be positive")
        if (warmupUpdates !in 0 until totalUpdates) {
            throw TrainingException("warmupUpdates must be in [0, totalUpdates)")
        }
    }

    override fun getNewValue(numUpdate: Int): Float =
        learningRateForUpdate(numUpdate.coerceIn(1, totalUpdates))

    fun learningRateForUpdate(updateNumber: Int): Float {
        if (updateNumber !in 1..totalUpdates) {
            throw TrainingException("updateNumber must be in [1, $totalUpdates]")
        }
        if (warmupUpdates > 0 && updateNumber <= warmupUpdates) {
            return maximumLearningRate * updateNumber / warmupUpdates
        }

        val decayUpdateCount = totalUpdates - warmupUpdates
        if (decayUpdateCount <= 1) return maximumLearningRate
        val decayIndex = updateNumber - warmupUpdates - 1
        val progress = decayIndex.toDouble() / (decayUpdateCount - 1)
        val cosineFactor = (1.0 + cos(PI * progress)) / 2.0
        return (minimumLearningRate + (maximumLearningRate - minimumLearningRate) * cosineFactor).toFloat()
    }
}
