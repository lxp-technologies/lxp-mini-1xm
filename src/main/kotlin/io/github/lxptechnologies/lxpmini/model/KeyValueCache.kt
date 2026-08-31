package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager

class DecoderKeyValueCache internal constructor(
    parentManager: NDManager,
    val layerCount: Int,
    val headCount: Int,
    val headDimension: Int,
    val maximumSequenceLength: Int,
) : AutoCloseable {
    private val cacheManager = parentManager.newSubManager()
    internal val layers = List(layerCount) {
        AttentionKeyValueCache(cacheManager, headCount, headDimension, maximumSequenceLength)
    }
    var tokenCount: Int = 0
        private set

    val isOpen: Boolean
        get() = cacheManager.isOpen

    init {
        cacheManager.name = "decoder-kv-cache"
    }

    internal fun advance(count: Int) {
        if (count <= 0 || tokenCount + count > maximumSequenceLength) {
            throw TensorShapeException(
                "Cannot advance KV cache from $tokenCount by $count with context $maximumSequenceLength",
            )
        }
        tokenCount += count
    }

    fun clear() {
        requireOpen()
        layers.forEach(AttentionKeyValueCache::clear)
        tokenCount = 0
    }

    override fun close() {
        if (cacheManager.isOpen) cacheManager.close()
    }

    private fun requireOpen() {
        if (!cacheManager.isOpen) throw TensorShapeException("KV cache is closed")
    }
}

internal class AttentionKeyValueCache(
    private val manager: NDManager,
    private val headCount: Int,
    private val headDimension: Int,
    private val maximumSequenceLength: Int,
) {
    private var cachedKeys: NDArray? = null
    private var cachedValues: NDArray? = null

    val tokenCount: Int
        get() = cachedKeys?.shape?.get(2)?.toInt() ?: 0

    fun append(keys: NDArray, values: NDArray): CachedKeyValues {
        requireCompatible(keys, values)
        val existingKeys = cachedKeys
        val existingValues = cachedValues
        if (existingKeys == null || existingValues == null) {
            keys.attach(manager)
            values.attach(manager)
            cachedKeys = keys
            cachedValues = values
            return CachedKeyValues(keys, values)
        }

        var combinedKeys: NDArray? = null
        var combinedValues: NDArray? = null
        try {
            combinedKeys = existingKeys.concat(keys, 2)
            combinedValues = existingValues.concat(values, 2)
            if (combinedKeys.manager !== manager) combinedKeys.attach(manager)
            if (combinedValues.manager !== manager) combinedValues.attach(manager)
            cachedKeys = combinedKeys
            cachedValues = combinedValues
            existingKeys.close()
            existingValues.close()
            return CachedKeyValues(combinedKeys, combinedValues)
        } catch (throwable: Throwable) {
            combinedKeys?.close()
            combinedValues?.close()
            throw throwable
        }
    }

    fun clear() {
        cachedKeys?.close()
        cachedValues?.close()
        cachedKeys = null
        cachedValues = null
    }

    private fun requireCompatible(keys: NDArray, values: NDArray) {
        if (keys.shape.dimension() != 4 || values.shape != keys.shape) {
            throw TensorShapeException("KV cache expects matching [B, H, T, D] keys and values")
        }
        if (keys.shape[0] <= 0 || keys.shape[1] != headCount.toLong() || keys.shape[3] != headDimension.toLong()
        ) {
            throw TensorShapeException(
                "KV cache expects [B, $headCount, T, $headDimension], got ${keys.shape}",
            )
        }
        if (keys.shape[2] <= 0 || tokenCount + keys.shape[2] > maximumSequenceLength) {
            throw TensorShapeException(
                "KV cache length ${tokenCount + keys.shape[2]} exceeds context $maximumSequenceLength",
            )
        }
        cachedKeys?.let { existing ->
            if (existing.shape[0] != keys.shape[0]) {
                throw TensorShapeException("KV cache batch size cannot change within a request")
            }
        }
    }
}

internal data class CachedKeyValues(val keys: NDArray, val values: NDArray)
