package io.github.lxptechnologies.lxpmini.server

import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test

class ChatPromptFormatterTest {
    private val formatter: ChatPromptFormatter = BaseModelChatPromptFormatter()

    @Test
    fun `base formatter serializes roles as replaceable plain text`() {
        val prompt = formatter.format(
            "Be concise",
            listOf(
                PlaygroundMessage("user", "Hello"),
                PlaygroundMessage("assistant", "Hi"),
            ),
            "Continue",
        )

        assertThat(prompt).isEqualTo(
            "System: Be concise\nUser: Hello\nAssistant: Hi\nUser: Continue\nAssistant:",
        )
    }

    @Test
    fun `formatter rejects roles the base serialization does not define`() {
        assertThatThrownBy {
            formatter.format(null, listOf(PlaygroundMessage("system", "hidden")), "hello")
        }.isInstanceOf(OpenAiApiException::class.java)
    }
}
