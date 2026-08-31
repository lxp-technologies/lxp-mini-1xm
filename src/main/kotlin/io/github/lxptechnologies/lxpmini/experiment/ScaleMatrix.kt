package io.github.lxptechnologies.lxpmini.experiment

import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.kotlin.KotlinModule
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.config.ModelConfig
import io.github.lxptechnologies.lxpmini.config.ProjectConfig
import io.github.lxptechnologies.lxpmini.model.ParameterCount
import io.github.lxptechnologies.lxpmini.model.ParameterCounter
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale

class ScaleMatrixLoader(
    private val configLoader: ConfigLoader = ConfigLoader(),
    private val parameterCounter: ParameterCounter = ParameterCounter(),
    private val mapper: ObjectMapper = matrixMapper(),
) {
    fun load(path: Path): ValidatedScaleMatrix {
        if (!Files.isRegularFile(path)) throw ScaleExperimentException("Scale matrix does not exist: $path")
        val definition = try {
            Files.newBufferedReader(path).use { reader -> mapper.readValue(reader, ScaleMatrixDefinition::class.java) }
        } catch (exception: JsonProcessingException) {
            throw ScaleExperimentException("Invalid scale matrix $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw ScaleExperimentException("Cannot read scale matrix $path: ${exception.message}", exception)
        }
        validateDefinition(definition)
        val baseDirectory = path.toAbsolutePath().parent
        val variants = definition.variants.map { variant ->
            val dimension = ScaleDimension.parse(variant.dimension)
            val configPath = baseDirectory.resolve(variant.config).normalize()
            val config = configLoader.load(configPath)
            ScaleVariant(variant.name, dimension, configPath, config, parameterCounter.count(config.model))
        }
        val baseline = variants.singleOrNull { it.name == definition.baseline }
            ?: throw ScaleExperimentException("Baseline '${definition.baseline}' must name exactly one variant")
        if (baseline.dimension != ScaleDimension.BASELINE) {
            throw ScaleExperimentException("Baseline '${baseline.name}' must declare dimension 'baseline'")
        }
        variants.forEach { variant -> validateAxis(baseline, variant) }
        return ValidatedScaleMatrix(definition.name, baseline.name, variants)
    }

    private fun validateDefinition(definition: ScaleMatrixDefinition) {
        if (definition.version != FORMAT_VERSION) {
            throw ScaleExperimentException("Scale matrix version must be $FORMAT_VERSION")
        }
        if (definition.name.isBlank()) throw ScaleExperimentException("Scale matrix name cannot be blank")
        if (definition.baseline.isBlank()) throw ScaleExperimentException("Scale matrix baseline cannot be blank")
        if (definition.variants.size < 2) throw ScaleExperimentException("Scale matrix must contain at least two variants")
        definition.variants.forEach { variant ->
            if (!variant.name.matches(VARIANT_NAME_PATTERN)) {
                throw ScaleExperimentException("Invalid variant name '${variant.name}'")
            }
            if (variant.config.isBlank()) throw ScaleExperimentException("Variant '${variant.name}' config cannot be blank")
        }
        val duplicate = definition.variants.groupingBy(ScaleVariantDefinition::name).eachCount()
            .entries.firstOrNull { it.value > 1 }?.key
        if (duplicate != null) throw ScaleExperimentException("Duplicate variant name '$duplicate'")
    }

    private fun validateAxis(baseline: ScaleVariant, variant: ScaleVariant) {
        if (variant.config.training != baseline.config.training) {
            throw ScaleExperimentException("Variant '${variant.name}' changes training configuration")
        }
        val model = variant.config.model
        val control = baseline.config.model
        when (variant.dimension) {
            ScaleDimension.BASELINE -> {
                if (variant.name != baseline.name || model != control) {
                    throw ScaleExperimentException("Only '${baseline.name}' may be an unchanged baseline")
                }
            }

            ScaleDimension.WIDTH -> {
                requireOnly(
                    variant,
                    model.copy(dModel = control.dModel, numHeads = control.numHeads) == control,
                    "dModel and numHeads",
                )
                if (model.dModel == control.dModel || model.headDim != control.headDim) {
                    throw ScaleExperimentException(
                        "Width variant '${variant.name}' must change dModel while preserving headDim ${control.headDim}",
                    )
                }
            }

            ScaleDimension.DEPTH -> {
                requireOnly(variant, model.copy(numLayers = control.numLayers) == control, "numLayers")
                if (model.numLayers == control.numLayers) {
                    throw ScaleExperimentException("Depth variant '${variant.name}' must change numLayers")
                }
            }

            ScaleDimension.CONTEXT -> {
                requireOnly(variant, model.copy(contextLength = control.contextLength) == control, "contextLength")
                if (model.contextLength == control.contextLength) {
                    throw ScaleExperimentException("Context variant '${variant.name}' must change contextLength")
                }
            }
        }
    }

    private fun requireOnly(variant: ScaleVariant, valid: Boolean, allowed: String) {
        if (!valid) {
            throw ScaleExperimentException("Variant '${variant.name}' may change only $allowed from the baseline")
        }
    }

    companion object {
        const val FORMAT_VERSION = 1
        private val VARIANT_NAME_PATTERN = Regex("[a-z0-9][a-z0-9-]{0,63}")

        private fun matrixMapper(): ObjectMapper = ObjectMapper(YAMLFactory())
            .registerModule(KotlinModule.Builder().build())
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
    }
}

enum class ScaleDimension {
    BASELINE,
    WIDTH,
    DEPTH,
    CONTEXT;

    companion object {
        fun parse(value: String): ScaleDimension = try {
            valueOf(value.uppercase(Locale.ROOT))
        } catch (_: IllegalArgumentException) {
            throw ScaleExperimentException("Unknown scale dimension '$value'; use baseline, width, depth or context")
        }
    }
}

@JsonPropertyOrder("version", "name", "baseline", "variants")
data class ScaleMatrixDefinition(
    val version: Int = ScaleMatrixLoader.FORMAT_VERSION,
    val name: String,
    val baseline: String,
    val variants: List<ScaleVariantDefinition>,
)

@JsonPropertyOrder("name", "dimension", "config")
data class ScaleVariantDefinition(
    val name: String,
    val dimension: String,
    val config: String,
)

data class ValidatedScaleMatrix(
    val name: String,
    val baseline: String,
    val variants: List<ScaleVariant>,
)

data class ScaleVariant(
    val name: String,
    val dimension: ScaleDimension,
    val configPath: Path,
    val config: ProjectConfig,
    val parameters: ParameterCount,
) {
    val model: ModelConfig
        get() = config.model
}
