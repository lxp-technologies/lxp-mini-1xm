package io.github.lxptechnologies.lxpmini.model

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import io.github.lxptechnologies.lxpmini.config.ModelConfig

class DecoderLanguageModel(
    parentManager: NDManager,
    config: ModelConfig,
) : AbstractBlock(VERSION), AutoCloseable {
    val config: ModelConfig = requireSupportedConfig(config)
    val embedding: TokenEmbedding = addChildBlock(
        "embedding",
        TokenEmbedding(this.config.vocabSize, this.config.dModel),
    )
    val transformerBlocks: List<TransformerBlock> = List(this.config.numLayers) { index ->
        addChildBlock(
            "block$index",
            TransformerBlock(
                parentManager,
                this.config.dModel,
                this.config.numHeads,
                this.config.ffnDim,
                this.config.contextLength,
                this.config.ropeTheta,
            ),
        )
    }
    val finalNorm: RmsNorm = addChildBlock("finalNorm", RmsNorm(this.config.dModel))
    val languageModelHead: LanguageModelHead = addChildBlock(
        "languageModelHead",
        LanguageModelHead(
            embedding.weightParameter,
            this.config.dModel,
            this.config.vocabSize,
            this.config.tieEmbeddings,
        ),
    )

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList = NDList(forwardWithIntermediates(parameterStore, inputs.singletonOrThrow(), training).logits)

    fun forwardWithIntermediates(
        parameterStore: ParameterStore,
        tokenIds: NDArray,
        training: Boolean,
    ): DecoderLanguageModelResult {
        requireTokenShape(tokenIds.shape)
        val embedded = embedding.forward(parameterStore, NDList(tokenIds), training).singletonOrThrow()
        val blockOutputs = ArrayList<NDArray>(transformerBlocks.size)
        var hiddenState = embedded
        for (block in transformerBlocks) {
            hiddenState = block.forward(parameterStore, NDList(hiddenState), training).singletonOrThrow()
            blockOutputs += hiddenState
        }
        val normalized = finalNorm.forward(parameterStore, NDList(hiddenState), training).singletonOrThrow()
        val logits = languageModelHead.forward(parameterStore, NDList(normalized), training).singletonOrThrow()
        return DecoderLanguageModelResult(logits, embedded, blockOutputs, normalized)
    }

    fun newKeyValueCache(parentManager: NDManager): DecoderKeyValueCache = DecoderKeyValueCache(
        parentManager = parentManager,
        layerCount = config.numLayers,
        headCount = config.numHeads,
        headDimension = config.headDim,
        maximumSequenceLength = config.contextLength,
    )

    fun forwardIncremental(
        parameterStore: ParameterStore,
        tokenIds: NDArray,
        cache: DecoderKeyValueCache,
    ): NDArray {
        requireTokenShape(tokenIds.shape)
        requireCompatibleCache(cache, tokenIds.shape[1].toInt())
        var hiddenState = embedding.forward(parameterStore, NDList(tokenIds), false).singletonOrThrow()
        for (index in transformerBlocks.indices) {
            hiddenState = transformerBlocks[index].forwardIncremental(
                parameterStore,
                hiddenState,
                cache.layers[index],
            )
        }
        val normalized = finalNorm.forward(parameterStore, NDList(hiddenState), false).singletonOrThrow()
        val logits = languageModelHead.forward(parameterStore, NDList(normalized), false).singletonOrThrow()
        cache.advance(tokenIds.shape[1].toInt())
        return logits
    }

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        if (inputShapes.size != 1) throw TensorShapeException("DecoderLanguageModel expects exactly one input shape")
        requireTokenShape(inputShapes[0])
        val input = inputShapes[0]
        return arrayOf(Shape(input[0], input[1], config.vocabSize.toLong()))
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        if (inputShapes.size != 1) throw TensorShapeException("DecoderLanguageModel expects exactly one input shape")
        requireTokenShape(inputShapes[0])
        val tokenShape = inputShapes[0]
        val hiddenShape = Shape(tokenShape[0], tokenShape[1], config.dModel.toLong())
        embedding.initialize(manager, dataType, tokenShape)
        transformerBlocks.forEach { it.initialize(manager, dataType, hiddenShape) }
        finalNorm.initialize(manager, dataType, hiddenShape)
        languageModelHead.initialize(manager, dataType, hiddenShape)
    }

    fun actualParameterCount(): Long {
        check(isInitialized) { "DecoderLanguageModel must be initialized before counting actual parameters" }
        return getParameters().values().sumOf { it.array.size() }
    }

    fun parameterTensorCount(): Int = getParameters().size()

    fun sharesEmbeddingParameter(): Boolean =
        languageModelHead.weightParameter === embedding.weightParameter

    fun sharesEmbeddingArray(): Boolean =
        isInitialized && languageModelHead.weightParameter.array === embedding.weightParameter.array

    fun openRopeCacheCount(): Int = transformerBlocks.count { it.isRopeCacheOpen() }

    override fun close() {
        var failure: Throwable? = null
        try {
            clear()
        } catch (throwable: Throwable) {
            failure = throwable
        }
        for (block in transformerBlocks) {
            try {
                block.close()
            } catch (throwable: Throwable) {
                if (failure == null) failure = throwable else failure.addSuppressed(throwable)
            }
        }
        failure?.let { throw it }
    }

    private fun requireTokenShape(shape: Shape) {
        if (shape.dimension() != EXPECTED_INPUT_RANK) {
            throw TensorShapeException("DecoderLanguageModel expects token IDs [B, T], got $shape")
        }
        if (shape[0] <= 0 || shape[1] <= 0) {
            throw TensorShapeException("DecoderLanguageModel requires positive batch and sequence dimensions, got $shape")
        }
        if (shape[1] > config.contextLength) {
            throw TensorShapeException("Sequence length ${shape[1]} exceeds context ${config.contextLength}")
        }
    }

    private fun requireCompatibleCache(cache: DecoderKeyValueCache, newTokenCount: Int) {
        if (!cache.isOpen) throw TensorShapeException("KV cache is closed")
        if (cache.layerCount != config.numLayers || cache.headCount != config.numHeads ||
            cache.headDimension != config.headDim || cache.maximumSequenceLength != config.contextLength
        ) {
            throw TensorShapeException("KV cache architecture does not match the decoder")
        }
        if (cache.tokenCount + newTokenCount > config.contextLength) {
            throw TensorShapeException(
                "KV cache length ${cache.tokenCount + newTokenCount} exceeds context ${config.contextLength}",
            )
        }
        if (cache.layers.any { layer -> layer.tokenCount != cache.tokenCount }) {
            throw TensorShapeException("KV cache layers are not synchronized")
        }
    }

    private companion object {
        const val VERSION: Byte = 1
        const val EXPECTED_INPUT_RANK = 2

        fun requireSupportedConfig(config: ModelConfig): ModelConfig {
            if (config.vocabSize <= 0) throw TensorShapeException("vocabSize must be positive")
            if (config.contextLength <= 0) throw TensorShapeException("contextLength must be positive")
            if (config.dModel <= 0) throw TensorShapeException("dModel must be positive")
            if (config.numLayers <= 0) throw TensorShapeException("numLayers must be positive")
            if (config.numHeads <= 0) throw TensorShapeException("numHeads must be positive")
            if (config.ffnDim <= 0) throw TensorShapeException("ffnDim must be positive")
            if (config.dModel % config.numHeads != 0) {
                throw TensorShapeException("dModel must be divisible by numHeads")
            }
            if (config.headDim % 2 != 0) throw TensorShapeException("headDim must be even for RoPE")
            if (!config.ropeTheta.isFinite() || config.ropeTheta <= 0.0) {
                throw TensorShapeException("ropeTheta must be finite and positive")
            }
            if (config.dropout != 0.0) {
                throw TensorShapeException("dropout is not implemented yet and must be 0.0")
            }
            return config
        }
    }
}

data class DecoderLanguageModelResult(
    val logits: NDArray,
    val embeddings: NDArray,
    val blockOutputs: List<NDArray>,
    val normalizedOutput: NDArray,
)
