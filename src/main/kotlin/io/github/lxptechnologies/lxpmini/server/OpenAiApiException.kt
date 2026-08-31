package io.github.lxptechnologies.lxpmini.server

import org.springframework.http.HttpStatus

class OpenAiApiException(
    val status: HttpStatus,
    message: String,
    val type: String = "invalid_request_error",
    val param: String? = null,
    val code: String,
    cause: Throwable? = null,
) : RuntimeException(message, cause)
