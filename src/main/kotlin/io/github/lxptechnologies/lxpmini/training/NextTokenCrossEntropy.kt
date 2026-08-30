package io.github.lxptechnologies.lxpmini.training

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.training.loss.SoftmaxCrossEntropyLoss

class NextTokenCrossEntropy {
    private val loss = SoftmaxCrossEntropyLoss(NAME, 1f, CLASS_AXIS, true, true)

    fun evaluate(targetIds: NDArray, logits: NDArray): NDArray {
        requireShapes(targetIds, logits)
        val tokenCount = logits.shape[0] * logits.shape[1]
        val vocabularySize = logits.shape[2]
        val flatTargets = targetIds.reshape(tokenCount)
        val flatLogits = logits.reshape(tokenCount, vocabularySize)
        return loss.evaluate(NDList(flatTargets), NDList(flatLogits))
    }

    private fun requireShapes(targetIds: NDArray, logits: NDArray) {
        if (logits.shape.dimension() != LOGITS_RANK) {
            throw TrainingException("Logits must have shape [B, T, V], got ${logits.shape}")
        }
        if (targetIds.shape.dimension() != TARGET_RANK) {
            throw TrainingException("Targets must have shape [B, T], got ${targetIds.shape}")
        }
        if (targetIds.shape[0] != logits.shape[0] || targetIds.shape[1] != logits.shape[1]) {
            throw TrainingException("Targets ${targetIds.shape} must match logits batch and sequence ${logits.shape}")
        }
        if (logits.shape[2] <= 1) throw TrainingException("Vocabulary dimension must be greater than one")
        if (targetIds.device != logits.device) {
            throw TrainingException("Targets and logits must be on the same device")
        }
    }

    private companion object {
        const val NAME = "NextTokenCrossEntropy"
        const val CLASS_AXIS = -1
        const val LOGITS_RANK = 3
        const val TARGET_RANK = 2
    }
}
