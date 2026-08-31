package io.github.lxptechnologies.lxpmini.config

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import kotlin.io.path.writeText

class ConfigLoaderTest {
    private val loader = ConfigLoader()

    @Test
    fun `loads and validates the reference preset`() {
        val config = loader.load(Path.of("configs/mini-17m.yaml"))

        assertThat(config.model.dModel).isEqualTo(384)
        assertThat(config.model.headDim).isEqualTo(64)
        assertThat(config.training.seed).isEqualTo(42)
        assertThat(config.runtime.device).isEqualTo("auto")
    }

    @Test
    fun `rejects unknown YAML properties`(@TempDir directory: Path) {
        val path = directory.resolve("unknown-property.yaml")
        path.writeText(validYaml().replace("  dModel: 384", "  dModel: 384\n  hiddenMagic: true"))

        assertThatThrownBy { loader.load(path) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("hiddenMagic")
    }

    @Test
    fun `explains when the configuration file is missing`(@TempDir directory: Path) {
        val missing = directory.resolve("missing.yaml")

        assertThatThrownBy { loader.load(missing) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessage("Configuration file does not exist: $missing")
    }

    @Test
    fun `validates an explicit runtime device`(@TempDir directory: Path) {
        val valid = directory.resolve("cpu.yaml")
        valid.writeText(validYaml() + "\nruntime:\n  device: cpu")
        assertThat(loader.load(valid).runtime.device).isEqualTo("cpu")

        val invalid = directory.resolve("invalid-device.yaml")
        invalid.writeText(validYaml() + "\nruntime:\n  device: cuda")
        assertThatThrownBy { loader.load(invalid) }
            .isInstanceOf(ConfigException::class.java)
            .hasMessageContaining("runtime.device must be one of")
    }

    private fun validYaml() = """
        model:
          vocabSize: 8192
          contextLength: 256
          dModel: 384
          numLayers: 8
          numHeads: 6
          ffnDim: 1024
          ropeTheta: 10000.0
          dropout: 0.0
          tieEmbeddings: true
        training:
          batchSize: 16
          gradientAccumulationSteps: 4
          learningRate: 0.0003
          minLearningRate: 0.00003
          warmupSteps: 500
          weightDecay: 0.1
          beta1: 0.9
          beta2: 0.95
          gradientClipNorm: 1.0
          seed: 42
    """.trimIndent()
}
