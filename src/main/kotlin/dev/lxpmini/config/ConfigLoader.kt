package dev.lxpmini.config

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.KotlinModule
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

class ConfigLoader(
    private val validator: ConfigValidator = ConfigValidator(),
    private val mapper: ObjectMapper = defaultMapper(),
) {
    fun load(path: Path): ProjectConfig {
        if (!Files.isRegularFile(path)) {
            throw ConfigException("Configuration file does not exist: $path")
        }

        val config = try {
            Files.newBufferedReader(path).use { reader ->
                mapper.readValue(reader, ProjectConfig::class.java)
            }
        } catch (exception: JsonProcessingException) {
            throw ConfigException("Invalid YAML in $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw ConfigException("Cannot read configuration $path: ${exception.message}", exception)
        }

        return validator.validate(config)
    }

    companion object {
        private fun defaultMapper(): ObjectMapper = ObjectMapper(YAMLFactory())
            .registerModule(KotlinModule.Builder().build())
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
    }
}

class ConfigException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
