package dev.lxpmini.model

import dev.lxpmini.config.ModelConfig
import java.text.NumberFormat
import java.util.Locale

class ParameterCounter {
    fun count(config: ModelConfig): ParameterCount {
        val tokenEmbeddings = config.vocabSize.toLong() * config.dModel
        val attention = config.numLayers.toLong() * 4 * config.dModel * config.dModel
        val feedForward = config.numLayers.toLong() * 3 * config.dModel * config.ffnDim
        val normalization = config.numLayers.toLong() * 2 * config.dModel + config.dModel
        val outputHead = if (config.tieEmbeddings) 0 else tokenEmbeddings

        return ParameterCount(
            tokenEmbeddings = tokenEmbeddings,
            attention = attention,
            feedForward = feedForward,
            normalization = normalization,
            outputHead = outputHead,
        )
    }
}

data class ParameterCount(
    val tokenEmbeddings: Long,
    val attention: Long,
    val feedForward: Long,
    val normalization: Long,
    val outputHead: Long,
) {
    val total: Long
        get() = tokenEmbeddings + attention + feedForward + normalization + outputHead

    fun format(config: ModelConfig): String {
        fun Long.display(): String = NUMBER_FORMAT.format(this)
        fun Int.display(): String = NUMBER_FORMAT.format(this)

        return buildString {
            appendLine("Model configuration")
            appendLine()
            appendLine("Vocabulary:           ${config.vocabSize.display()}")
            appendLine("Context length:       ${config.contextLength.display()}")
            appendLine("Embedding size:       ${config.dModel.display()}")
            appendLine("Layers:               ${config.numLayers.display()}")
            appendLine("Attention heads:      ${config.numHeads.display()}")
            appendLine("Head dimension:       ${config.headDim.display()}")
            appendLine("FFN dimension:        ${config.ffnDim.display()}")
            appendLine()
            appendLine("Parameter estimate (bias-free linear projections)")
            appendLine()
            appendLine("Token embeddings:     ${tokenEmbeddings.display()}")
            appendLine("Attention:            ${attention.display()}")
            appendLine("Feed-forward:         ${feedForward.display()}")
            appendLine("Normalization:        ${normalization.display()}")
            appendLine(
                if (config.tieEmbeddings) "Output head:          tied (0 additional)"
                else "Output head:          ${outputHead.display()}",
            )
            appendLine()
            append("Total parameters:     ${total.display()}")
        }
    }

    private companion object {
        val NUMBER_FORMAT: NumberFormat = NumberFormat.getIntegerInstance(Locale.CANADA)
    }
}
