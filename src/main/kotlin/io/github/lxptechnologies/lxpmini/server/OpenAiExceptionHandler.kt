package io.github.lxptechnologies.lxpmini.server

import com.fasterxml.jackson.databind.exc.UnrecognizedPropertyException
import io.github.lxptechnologies.lxpmini.inference.InferenceException
import org.slf4j.LoggerFactory
import org.springframework.http.HttpStatus
import org.springframework.http.ResponseEntity
import org.springframework.http.converter.HttpMessageNotReadableException
import org.springframework.web.bind.annotation.ExceptionHandler
import org.springframework.web.bind.annotation.RestControllerAdvice
import org.springframework.web.servlet.resource.NoResourceFoundException

@RestControllerAdvice
class OpenAiExceptionHandler {
    @ExceptionHandler(OpenAiApiException::class)
    fun apiError(exception: OpenAiApiException): ResponseEntity<OpenAiErrorResponse> = response(
        exception.status,
        exception.message ?: "Invalid request",
        exception.type,
        exception.param,
        exception.code,
    )

    @ExceptionHandler(HttpMessageNotReadableException::class)
    fun malformedJson(exception: HttpMessageNotReadableException): ResponseEntity<OpenAiErrorResponse> {
        val unknown = exception.findCause<UnrecognizedPropertyException>()
        return if (unknown != null) {
            response(
                HttpStatus.BAD_REQUEST,
                "Unknown field '${unknown.propertyName}'",
                "invalid_request_error",
                unknown.propertyName,
                "unknown_parameter",
            )
        } else {
            response(HttpStatus.BAD_REQUEST, "Malformed JSON request", "invalid_request_error", null, "invalid_json")
        }
    }

    @ExceptionHandler(InferenceException::class)
    fun inferenceError(exception: InferenceException): ResponseEntity<OpenAiErrorResponse> {
        if (exception.message?.contains("exceeds context") == true) {
            return response(
                HttpStatus.BAD_REQUEST,
                exception.message ?: "Context length exceeded",
                "invalid_request_error",
                "max_tokens",
                "context_length_exceeded",
            )
        }
        logger.error("Inference request failed", exception)
        return response(HttpStatus.INTERNAL_SERVER_ERROR, "Inference failed", "server_error", null, "inference_error")
    }

    @ExceptionHandler(NoResourceFoundException::class)
    fun notFound(): ResponseEntity<OpenAiErrorResponse> =
        response(HttpStatus.NOT_FOUND, "Endpoint not found", "invalid_request_error", null, "not_found")

    @ExceptionHandler(Exception::class)
    fun unexpected(exception: Exception): ResponseEntity<OpenAiErrorResponse> {
        logger.error("Unexpected HTTP adapter failure", exception)
        return response(HttpStatus.INTERNAL_SERVER_ERROR, "Internal server error", "server_error", null, "internal_error")
    }

    private fun response(
        status: HttpStatus,
        message: String,
        type: String,
        param: String?,
        code: String,
    ): ResponseEntity<OpenAiErrorResponse> = ResponseEntity
        .status(status)
        .body(OpenAiErrorResponse(OpenAiError(message, type, param, code)))

    private inline fun <reified T : Throwable> Throwable.findCause(): T? {
        var current: Throwable? = this
        while (current != null) {
            if (current is T) return current
            current = current.cause
        }
        return null
    }

    private companion object {
        val logger = LoggerFactory.getLogger(OpenAiExceptionHandler::class.java)
    }
}
