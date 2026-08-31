package io.github.lxptechnologies.lxpmini.config

data class ProjectConfig(
    val model: ModelConfig,
    val training: TrainingConfig,
    val runtime: RuntimeConfig = RuntimeConfig(),
)

data class RuntimeConfig(val device: String = "auto")

data class ModelConfig(
    val vocabSize: Int,
    val contextLength: Int,
    val dModel: Int,
    val numLayers: Int,
    val numHeads: Int,
    val ffnDim: Int,
    val ropeTheta: Double,
    val dropout: Double,
    val tieEmbeddings: Boolean,
) {
    val headDim: Int
        get() = dModel / numHeads
}

data class TrainingConfig(
    val batchSize: Int,
    val gradientAccumulationSteps: Int,
    val learningRate: Double,
    val minLearningRate: Double,
    val warmupSteps: Int,
    val weightDecay: Double,
    val beta1: Double,
    val beta2: Double,
    val gradientClipNorm: Double,
    val seed: Long,
)
