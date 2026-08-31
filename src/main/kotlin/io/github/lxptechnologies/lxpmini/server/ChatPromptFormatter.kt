package io.github.lxptechnologies.lxpmini.server

interface ChatPromptFormatter {
    fun format(systemPrompt: String?, messages: List<PlaygroundMessage>, userMessage: String): String
}

class BaseModelChatPromptFormatter : ChatPromptFormatter {
    override fun format(
        systemPrompt: String?,
        messages: List<PlaygroundMessage>,
        userMessage: String,
    ): String = buildString {
        systemPrompt?.takeIf(String::isNotBlank)?.let { prompt ->
            append("System: ").append(prompt.trim()).append('\n')
        }
        messages.forEach { message ->
            append(message.role.displayName).append(": ").append(message.content.trim()).append('\n')
        }
        append("User: ").append(userMessage.trim()).append('\n')
        append("Assistant:")
    }
}

private val String.displayName: String
    get() = when (this) {
        "user" -> "User"
        "assistant" -> "Assistant"
        else -> throw OpenAiApiException(
            org.springframework.http.HttpStatus.BAD_REQUEST,
            "Playground message role must be 'user' or 'assistant'",
            param = "messages.role",
            code = "invalid_value",
        )
    }
