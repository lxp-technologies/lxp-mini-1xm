package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

class RotaryPositionEmbedding(
    parentManager: NDManager,
    headDimension: Int,
    maximumSequenceLength: Int,
    theta: Double = DEFAULT_THETA,
) : AutoCloseable {
    val headDimension: Int = requireEvenHeadDimension(headDimension)
    val maximumSequenceLength: Int = requirePositiveSequenceLength(maximumSequenceLength)
    val theta: Double = requirePositiveTheta(theta)
    private val pairCount: Int = this.headDimension / 2
    private val cacheElementCount = requireCacheSize(this.maximumSequenceLength, pairCount)
    private val caches = createCaches(
        parentManager,
        this.headDimension,
        this.maximumSequenceLength,
        this.theta,
        pairCount,
        cacheElementCount,
    )
    private val cacheManager = caches.manager
    private val cosineCache: NDArray = caches.cosines
    private val sineCache: NDArray = caches.sines

    fun apply(input: NDArray, startPosition: Int = 0): NDArray {
        requireInputShape(input.shape, startPosition)
        if (input.device != cacheManager.device) {
            throw TensorShapeException(
                "RoPE input device ${input.device} differs from cache device ${cacheManager.device}",
            )
        }

        val batchSize = input.shape[0]
        val headCount = input.shape[1]
        val sequenceLength = input.shape[2]
        val paired = input.reshape(batchSize, headCount, sequenceLength, pairCount.toLong(), PAIR_SIZE)
        val even = paired.get(NDIndex("..., :, 0"))
        val odd = paired.get(NDIndex("..., :, 1"))
        val endPosition = startPosition + sequenceLength.toInt()
        val cosines = cosineCache.get(input.manager, NDIndex("$startPosition:$endPosition, :"))
        val sines = sineCache.get(input.manager, NDIndex("$startPosition:$endPosition, :"))
        val rotatedEven = even.mul(cosines).sub(odd.mul(sines))
        val rotatedOdd = even.mul(sines).add(odd.mul(cosines))
        return rotatedEven.stack(rotatedOdd, -1).reshape(input.shape)
    }

    fun cacheShape(): Shape = cosineCache.shape

    fun isOpen(): Boolean = cacheManager.isOpen

    override fun close() = cacheManager.close()

    private fun requireInputShape(shape: Shape, startPosition: Int) {
        if (shape.dimension() != EXPECTED_INPUT_RANK || shape.lastDimension != headDimension.toLong()) {
            throw TensorShapeException("RoPE expects [B, H, T, $headDimension], got $shape")
        }
        if (startPosition < 0 || startPosition.toLong() + shape[2] > maximumSequenceLength) {
            throw TensorShapeException(
                "Positions [$startPosition, ${startPosition.toLong() + shape[2]}) exceed context $maximumSequenceLength",
            )
        }
    }

    private companion object {
        const val EXPECTED_INPUT_RANK = 4
        const val PAIR_SIZE = 2L
        const val DEFAULT_THETA = 10_000.0

        fun requireEvenHeadDimension(value: Int): Int {
            if (value <= 0 || value % 2 != 0) {
                throw TensorShapeException("headDimension must be positive and even")
            }
            return value
        }

        fun requirePositiveSequenceLength(value: Int): Int {
            if (value <= 0) throw TensorShapeException("maximumSequenceLength must be positive")
            return value
        }

        fun requirePositiveTheta(value: Double): Double {
            if (!value.isFinite() || value <= 0.0) throw TensorShapeException("theta must be finite and positive")
            return value
        }

        fun requireCacheSize(sequenceLength: Int, pairCount: Int): Int {
            val elementCount = sequenceLength.toLong() * pairCount
            if (elementCount > Int.MAX_VALUE) throw TensorShapeException("RoPE cache exceeds the JVM array limit")
            return elementCount.toInt()
        }

        fun createCaches(
            parentManager: NDManager,
            headDimension: Int,
            sequenceLength: Int,
            theta: Double,
            pairCount: Int,
            elementCount: Int,
        ): RopeCaches {
            val cosineValues = FloatArray(elementCount)
            val sineValues = FloatArray(elementCount)
            for (position in 0 until sequenceLength) {
                for (pairIndex in 0 until pairCount) {
                    val inverseFrequency = theta.pow(-2.0 * pairIndex / headDimension)
                    val angle = position * inverseFrequency
                    val index = position * pairCount + pairIndex
                    cosineValues[index] = cos(angle).toFloat()
                    sineValues[index] = sin(angle).toFloat()
                }
            }

            val manager = parentManager.newSubManager()
            return try {
                val shape = Shape(sequenceLength.toLong(), pairCount.toLong())
                RopeCaches(
                    manager = manager,
                    cosines = manager.create(cosineValues, shape),
                    sines = manager.create(sineValues, shape),
                )
            } catch (throwable: Throwable) {
                manager.close()
                throw throwable
            }
        }
    }
}

private data class RopeCaches(
    val manager: NDManager,
    val cosines: NDArray,
    val sines: NDArray,
)
