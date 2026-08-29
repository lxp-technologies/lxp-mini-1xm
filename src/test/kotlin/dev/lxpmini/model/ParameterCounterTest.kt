package dev.lxpmini.model

import dev.lxpmini.config.ConfigLoader
import dev.lxpmini.config.validModelConfig
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import java.nio.file.Path

class ParameterCounterTest {
    private val counter = ParameterCounter()
    private val loader = ConfigLoader()

    @Test
    fun `counts every component of mini 17m`() {
        val count = counter.count(validModelConfig())

        assertThat(count.tokenEmbeddings).isEqualTo(3_145_728)
        assertThat(count.attention).isEqualTo(4_718_592)
        assertThat(count.feedForward).isEqualTo(9_437_184)
        assertThat(count.normalization).isEqualTo(6_528)
        assertThat(count.outputHead).isZero()
        assertThat(count.total).isEqualTo(17_308_032)
    }

    @Test
    fun `untied output head adds a second vocabulary projection`() {
        val tied = counter.count(validModelConfig())
        val untied = counter.count(validModelConfig().copy(tieEmbeddings = false))

        assertThat(untied.outputHead).isEqualTo(3_145_728)
        assertThat(untied.total - tied.total).isEqualTo(3_145_728)
        assertThat(untied.total).isEqualTo(20_453_760)
    }

    @Test
    fun `all named presets stay at their documented scale`() {
        val expected = mapOf(
            "mini-11m" to 10_981_440L,
            "mini-14m" to 13_767_552L,
            "mini-17m" to 17_308_032L,
            "mini-22m" to 22_618_752L,
        )

        expected.forEach { (preset, expectedTotal) ->
            val config = loader.load(Path.of("configs/$preset.yaml"))
            assertThat(counter.count(config.model).total)
                .describedAs("parameter count for %s", preset)
                .isEqualTo(expectedTotal)
        }
    }

    @Test
    fun `formatted report teaches dimensions and assumptions`() {
        val config = validModelConfig()

        val report = counter.count(config).format(config)

        assertThat(report).contains("Head dimension:       64")
        assertThat(report).contains("bias-free linear projections")
        assertThat(report).contains("Output head:          tied (0 additional)")
        assertThat(report).contains("Total parameters:     17,308,032")
    }
}
