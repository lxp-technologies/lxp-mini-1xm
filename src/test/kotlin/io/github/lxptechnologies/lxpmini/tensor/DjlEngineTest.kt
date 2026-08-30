package io.github.lxptechnologies.lxpmini.tensor

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.NDManager
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class DjlEngineTest {
    @Test
    fun `loads PyTorch and releases a manager scope`() {
        assertThat(Engine.getInstance().engineName).isEqualTo("PyTorch")

        NDManager.newBaseManager(Device.cpu()).use { manager ->
            val values = manager.create(floatArrayOf(1f, 2f, 3f))
            assertThat(values.sum().getFloat()).isEqualTo(6f)
        }
    }
}
