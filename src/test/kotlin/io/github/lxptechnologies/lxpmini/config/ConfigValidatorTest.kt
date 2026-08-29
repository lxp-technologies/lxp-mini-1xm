package io.github.lxptechnologies.lxpmini.config

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class ConfigValidatorTest {
    private val validator = ConfigValidator()

    @Test
    fun `accepts the mini 17m dimensions`() {
        val result = validator.validate(validProjectConfig())

        assertThat(result.model.headDim).isEqualTo(64)
    }

    @Test
    fun `rejects a model width that cannot be split evenly across heads`() {
        val invalid = validProjectConfig(
            model = validModelConfig().copy(dModel = 385),
        )

        assertThatThrownBy { validator.validate(invalid) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("model.dModel must be divisible by model.numHeads")
    }

    @Test
    fun `rejects an odd head dimension because RoPE rotates pairs`() {
        val invalid = validProjectConfig(
            model = validModelConfig().copy(dModel = 390),
        )

        assertThatThrownBy { validator.validate(invalid) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("model.headDim must be even")
    }

    @Test
    fun `reports every invalid field instead of stopping at the first`() {
        val invalid = validProjectConfig(
            model = validModelConfig().copy(vocabSize = 258, contextLength = 0),
            training = validTrainingConfig().copy(batchSize = 0, beta2 = 1.0),
        )

        assertThatThrownBy { validator.validate(invalid) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("model.vocabSize")
            .hasMessageContaining("model.contextLength")
            .hasMessageContaining("training.batchSize")
            .hasMessageContaining("training.beta2")
    }

    @Test
    fun `rejects a minimum learning rate above the initial rate`() {
        val invalid = validProjectConfig(
            training = validTrainingConfig().copy(
                learningRate = 0.0003,
                minLearningRate = 0.0004,
            ),
        )

        assertThatThrownBy { validator.validate(invalid) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("training.minLearningRate cannot exceed training.learningRate")
    }
}
