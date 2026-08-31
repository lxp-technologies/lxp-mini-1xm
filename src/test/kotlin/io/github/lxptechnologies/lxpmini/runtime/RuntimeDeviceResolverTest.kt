package io.github.lxptechnologies.lxpmini.runtime

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class RuntimeDeviceResolverTest {
    @Test
    fun `explicit cpu never selects an available GPU`() {
        val selection = resolver(gpuCount = 2).resolve("cpu")

        assertThat(selection.requested).isEqualTo(RuntimeDeviceRequest.CPU)
        assertThat(selection.selectedName).isEqualTo("cpu")
        assertThat(selection.selected.isGpu).isFalse()
    }

    @Test
    fun `auto selects GPU zero when available and CPU otherwise`() {
        assertThat(resolver(gpuCount = 2).resolve("auto").selectedName).isEqualTo("cuda:0")
        assertThat(resolver(gpuCount = 0).resolve("auto").selectedName).isEqualTo("cpu")
    }

    @Test
    fun `explicit CUDA fails instead of silently falling back`() {
        assertThatThrownBy { resolver(gpuCount = 0).resolve("cuda:0") }
            .isInstanceOf(RuntimeDeviceException::class.java)
            .hasMessageContaining("explicitly requested")
            .hasMessageContaining("no usable CUDA GPU")
    }

    @Test
    fun `selection reports exact engine diagnostics`() {
        val selection = resolver(gpuCount = 1).resolve("cuda:0")

        assertThat(selection.engineName).isEqualTo("PyTorch")
        assertThat(selection.djlVersion).isEqualTo("0.36.0")
        assertThat(selection.nativeRuntimeVersion).isEqualTo("2.7.1")
        assertThat(selection.gpuCount).isOne()
    }

    private fun resolver(gpuCount: Int) = RuntimeDeviceResolver {
        object : RuntimeEngine {
            override fun engineName() = "PyTorch"
            override fun djlVersion() = "0.36.0"
            override fun nativeRuntimeVersion() = "2.7.1"
            override fun gpuCount() = gpuCount
        }
    }
}
