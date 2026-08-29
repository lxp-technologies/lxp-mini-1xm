package dev.lxpmini.config

fun validProjectConfig(
    model: ModelConfig = validModelConfig(),
    training: TrainingConfig = validTrainingConfig(),
) = ProjectConfig(model = model, training = training)

fun validModelConfig() = ModelConfig(
    vocabSize = 8_192,
    contextLength = 256,
    dModel = 384,
    numLayers = 8,
    numHeads = 6,
    ffnDim = 1_024,
    ropeTheta = 10_000.0,
    dropout = 0.0,
    tieEmbeddings = true,
)

fun validTrainingConfig() = TrainingConfig(
    batchSize = 16,
    gradientAccumulationSteps = 4,
    learningRate = 0.0003,
    minLearningRate = 0.00003,
    warmupSteps = 500,
    weightDecay = 0.1,
    beta1 = 0.9,
    beta2 = 0.95,
    gradientClipNorm = 1.0,
    seed = 42,
)
