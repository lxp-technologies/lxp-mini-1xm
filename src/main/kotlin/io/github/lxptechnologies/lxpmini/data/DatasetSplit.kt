package io.github.lxptechnologies.lxpmini.data

import kotlin.math.floor

data class TokenRange(
    val startInclusive: Long,
    val endExclusive: Long,
) {
    val size: Long = endExclusive - startInclusive

    init {
        if (startInclusive < 0 || endExclusive < startInclusive) {
            throw DatasetException("Invalid token range [$startInclusive, $endExclusive)")
        }
    }
}

data class DatasetSplit(
    val train: TokenRange,
    val validation: TokenRange,
)

object DeterministicTokenSplit {
    fun contiguous(totalTokenCount: Long, validationFraction: Double): DatasetSplit {
        if (totalTokenCount < 0) throw DatasetException("totalTokenCount must be non-negative")
        if (!validationFraction.isFinite() || validationFraction !in 0.0..1.0) {
            throw DatasetException("validationFraction must be between 0.0 and 1.0")
        }

        val validationCount = floor(totalTokenCount * validationFraction).toLong()
        val boundary = totalTokenCount - validationCount
        return DatasetSplit(
            train = TokenRange(0, boundary),
            validation = TokenRange(boundary, totalTokenCount),
        )
    }
}
