package io.github.lxptechnologies.lxpmini.experiment

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class ScaleMatrixLoaderTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `loads the PR13 matrix with controlled parameter counts`() {
        val matrix = ScaleMatrixLoader().load(Path.of("configs/pr13/matrix.yaml"))

        assertThat(matrix.baseline).isEqualTo("baseline-14m")
        assertThat(matrix.variants.associate { it.name to it.parameters.total }).containsExactlyInAnyOrderEntriesOf(
            mapOf(
                "baseline-14m" to 14_266_752L,
                "width-11m" to 11_233_600L,
                "depth-18m" to 17_807_232L,
                "depth-21m" to 21_347_712L,
                "context-8" to 14_266_752L,
                "context-32" to 14_266_752L,
            ),
        )
        assertThat(matrix.variants.single { it.name == "width-11m" }.model.headDim).isEqualTo(64)
    }

    @Test
    fun `rejects a hidden context change declared as depth`() {
        val baseline = Path.of("configs/pr13/baseline-14m.yaml").toAbsolutePath().invariant()
        val changedContext = Path.of("configs/pr13/context-32.yaml").toAbsolutePath().invariant()
        val matrix = temporaryDirectory.resolve("invalid.yaml")
        Files.writeString(
            matrix,
            """
            version: 1
            name: invalid
            baseline: control
            variants:
              - name: control
                dimension: baseline
                config: $baseline
              - name: hidden-change
                dimension: depth
                config: $changedContext
            """.trimIndent(),
        )

        assertThatThrownBy { ScaleMatrixLoader().load(matrix) }
            .isInstanceOf(ScaleExperimentException::class.java)
            .hasMessageContaining("may change only numLayers")
    }

    private fun Path.invariant(): String = toString().replace('\\', '/')
}
