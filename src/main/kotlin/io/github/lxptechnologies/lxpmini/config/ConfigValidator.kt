package io.github.lxptechnologies.lxpmini.config

import io.github.lxptechnologies.lxpmini.runtime.RuntimeDeviceRequest

class ConfigValidator {
    fun validate(config: ProjectConfig): ProjectConfig {
        val errors = buildList {
            validateModel(config.model, this)
            validateTraining(config.training, this)
            if (config.runtime.device !in RuntimeDeviceRequest.entries.map { request -> request.value }) {
                add("runtime.device must be one of: auto, cpu, cuda:0")
            }
        }

        if (errors.isNotEmpty()) {
            throw ConfigException(errors.joinToString(separator = "; "))
        }
        return config
    }

    private fun validateModel(model: ModelConfig, errors: MutableList<String>) {
        if (model.vocabSize < MIN_BYTE_VOCAB_SIZE) {
            errors += "model.vocabSize must be at least $MIN_BYTE_VOCAB_SIZE (256 bytes + BOS/EOS/PAD)"
        }
        if (model.contextLength <= 0) errors += "model.contextLength must be positive"
        if (model.dModel <= 0) errors += "model.dModel must be positive"
        if (model.numLayers <= 0) errors += "model.numLayers must be positive"
        if (model.numHeads <= 0) errors += "model.numHeads must be positive"
        if (model.ffnDim <= 0) errors += "model.ffnDim must be positive"
        if (!model.ropeTheta.isFinite() || model.ropeTheta <= 0.0) {
            errors += "model.ropeTheta must be finite and positive"
        }
        if (!model.dropout.isFinite() || model.dropout !in 0.0..<1.0) {
            errors += "model.dropout must be in [0.0, 1.0)"
        }

        if (model.dModel > 0 && model.numHeads > 0) {
            if (model.dModel % model.numHeads != 0) {
                errors += "model.dModel must be divisible by model.numHeads"
            } else if (model.headDim % 2 != 0) {
                errors += "model.headDim must be even so RoPE can rotate pairs of dimensions"
            }
        }
    }

    private fun validateTraining(training: TrainingConfig, errors: MutableList<String>) {
        if (training.batchSize <= 0) errors += "training.batchSize must be positive"
        if (training.gradientAccumulationSteps <= 0) {
            errors += "training.gradientAccumulationSteps must be positive"
        }
        if (!training.learningRate.isFinite() || training.learningRate <= 0.0) {
            errors += "training.learningRate must be finite and positive"
        }
        if (!training.minLearningRate.isFinite() || training.minLearningRate < 0.0) {
            errors += "training.minLearningRate must be finite and non-negative"
        }
        if (training.minLearningRate > training.learningRate) {
            errors += "training.minLearningRate cannot exceed training.learningRate"
        }
        if (training.warmupSteps < 0) errors += "training.warmupSteps must be non-negative"
        if (!training.weightDecay.isFinite() || training.weightDecay < 0.0) {
            errors += "training.weightDecay must be finite and non-negative"
        }
        if (!training.beta1.isFinite() || training.beta1 !in 0.0..<1.0) {
            errors += "training.beta1 must be in [0.0, 1.0)"
        }
        if (!training.beta2.isFinite() || training.beta2 !in 0.0..<1.0) {
            errors += "training.beta2 must be in [0.0, 1.0)"
        }
        if (!training.gradientClipNorm.isFinite() || training.gradientClipNorm <= 0.0) {
            errors += "training.gradientClipNorm must be finite and positive"
        }
    }

    private companion object {
        const val MIN_BYTE_VOCAB_SIZE = 259
    }
}
