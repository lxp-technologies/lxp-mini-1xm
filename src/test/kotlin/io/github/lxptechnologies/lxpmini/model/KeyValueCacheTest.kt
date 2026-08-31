package io.github.lxptechnologies.lxpmini.model

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.assertj.core.data.Offset.offset
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock

@ResourceLock(value = "DJL_ENGINE", mode = ResourceAccessMode.READ_WRITE)
class KeyValueCacheTest {
    @Test
    fun `incremental logits match full causal forward within documented tolerance`() {
        Engine.getInstance().setRandomSeed(42)
        val tokens = intArrayOf(1, 4, 2, 7, 3, 9)
        NDManager.newBaseManager(Device.cpu()).use { root ->
            val modelManager = root.newSubManager()
            DecoderLanguageModel(modelManager, tinyConfig()).use { model ->
                model.initialize(modelManager, DataType.FLOAT32, Shape(1, tokens.size.toLong()))
                modelManager.cap()
                val parameterStore = ParameterStore(modelManager, false)
                val expected = fullLogits(root, model, parameterStore, tokens)

                root.newSubManager().use { requestManager ->
                    model.newKeyValueCache(requestManager).use { cache ->
                        val actual = ArrayList<FloatArray>()
                        tokens.forEach { token ->
                            requestManager.newSubManager().use { stepManager ->
                                val input = stepManager.create(longArrayOf(token.toLong()), Shape(1, 1))
                                actual += model.forwardIncremental(parameterStore, input, cache)
                                    .get("0, 0, :")
                                    .toFloatArray()
                            }
                        }

                        assertThat(cache.tokenCount).isEqualTo(tokens.size)
                        actual.forEachIndexed { position, logits ->
                            assertThat(logits).containsExactly(expected[position], offset(LOGIT_TOLERANCE))
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `prefill and interleaved request caches remain isolated and close independently`() {
        Engine.getInstance().setRandomSeed(7)
        val firstTokens = intArrayOf(1, 2, 3, 4)
        val secondTokens = intArrayOf(8, 7, 6, 5)
        NDManager.newBaseManager(Device.cpu()).use { root ->
            val modelManager = root.newSubManager()
            DecoderLanguageModel(modelManager, tinyConfig()).use { model ->
                model.initialize(modelManager, DataType.FLOAT32, Shape(1, 4))
                modelManager.cap()
                val parameterStore = ParameterStore(modelManager, false)
                val firstExpected = fullLogits(root, model, parameterStore, firstTokens).last()
                val secondExpected = fullLogits(root, model, parameterStore, secondTokens).last()
                val firstRequest = root.newSubManager()
                val secondRequest = root.newSubManager()
                val firstCache = model.newKeyValueCache(firstRequest)
                val secondCache = model.newKeyValueCache(secondRequest)
                try {
                    prefill(firstRequest, model, parameterStore, firstCache, firstTokens.take(3).toIntArray())
                    prefill(secondRequest, model, parameterStore, secondCache, secondTokens.take(3).toIntArray())
                    val firstActual = decode(firstRequest, model, parameterStore, firstCache, firstTokens.last())
                    val secondActual = decode(secondRequest, model, parameterStore, secondCache, secondTokens.last())

                    assertThat(firstActual).containsExactly(firstExpected, offset(LOGIT_TOLERANCE))
                    assertThat(secondActual).containsExactly(secondExpected, offset(LOGIT_TOLERANCE))
                    assertThat(firstCache.tokenCount).isEqualTo(4)
                    assertThat(secondCache.tokenCount).isEqualTo(4)

                    firstCache.close()
                    assertThat(firstCache.isOpen).isFalse()
                    assertThat(secondCache.isOpen).isTrue()
                    secondCache.clear()
                    assertThat(secondCache.tokenCount).isZero()
                    assertThatThrownBy { firstCache.clear() }
                        .isInstanceOf(TensorShapeException::class.java)
                        .hasMessageContaining("closed")
                } finally {
                    firstCache.close()
                    secondCache.close()
                    firstRequest.close()
                    secondRequest.close()
                }
            }
        }
    }

    private fun fullLogits(
        root: NDManager,
        model: DecoderLanguageModel,
        parameterStore: ParameterStore,
        tokens: IntArray,
    ): List<FloatArray> = root.newSubManager().use { temporary ->
        val input = temporary.create(tokens.map(Int::toLong).toLongArray(), Shape(1, tokens.size.toLong()))
        val logits = model.forward(parameterStore, NDList(input), false).singletonOrThrow()
        tokens.indices.map { position -> logits.get("0, $position, :").toFloatArray() }
    }

    private fun prefill(
        requestManager: NDManager,
        model: DecoderLanguageModel,
        parameterStore: ParameterStore,
        cache: DecoderKeyValueCache,
        tokens: IntArray,
    ) {
        requestManager.newSubManager().use { temporary ->
            val input = temporary.create(tokens.map(Int::toLong).toLongArray(), Shape(1, tokens.size.toLong()))
            model.forwardIncremental(parameterStore, input, cache).toFloatArray()
        }
    }

    private fun decode(
        requestManager: NDManager,
        model: DecoderLanguageModel,
        parameterStore: ParameterStore,
        cache: DecoderKeyValueCache,
        token: Int,
    ): FloatArray = requestManager.newSubManager().use { temporary ->
        val input = temporary.create(longArrayOf(token.toLong()), Shape(1, 1))
        model.forwardIncremental(parameterStore, input, cache).get("0, 0, :").toFloatArray()
    }

    private fun tinyConfig() = ModelConfig(
        vocabSize = 16,
        contextLength = 8,
        dModel = 8,
        numLayers = 2,
        numHeads = 2,
        ffnDim = 16,
        ropeTheta = 10_000.0,
        dropout = 0.0,
        tieEmbeddings = true,
    )

    private companion object {
        const val LOGIT_TOLERANCE = 1e-5f
    }
}
