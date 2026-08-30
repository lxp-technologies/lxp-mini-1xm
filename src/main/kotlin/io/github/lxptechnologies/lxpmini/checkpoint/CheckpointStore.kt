package io.github.lxptechnologies.lxpmini.checkpoint

import ai.djl.MalformedModelException
import ai.djl.ndarray.NDManager
import ai.djl.nn.Block
import com.fasterxml.jackson.annotation.JsonPropertyOrder
import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.SerializationFeature
import com.fasterxml.jackson.module.kotlin.KotlinModule
import io.github.lxptechnologies.lxpmini.training.TrainingProgress
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

class CheckpointStore(
    private val mapper: ObjectMapper = checkpointMapper(),
) {
    fun save(
        runDirectory: Path,
        model: Block,
        progress: TrainingProgress,
        totalUpdates: Int,
        configSha256: String,
    ): SavedCheckpoint {
        validateProgress(progress, totalUpdates)
        requireChecksum(configSha256, "configSha256")
        val checkpointId = checkpointId(progress.optimizerUpdates)
        val checkpointDirectory = runDirectory.resolve(CHECKPOINTS_DIRECTORY).resolve(checkpointId)
        if (Files.exists(checkpointDirectory)) {
            throw CheckpointException("Checkpoint already exists: $checkpointDirectory")
        }

        try {
            Files.createDirectories(checkpointDirectory)
            val modelPath = checkpointDirectory.resolve(MODEL_FILE)
            writeAtomically(modelPath) { temporaryPath ->
                DataOutputStream(BufferedOutputStream(Files.newOutputStream(temporaryPath))).use(model::saveParameters)
            }
            val modelSha256 = Sha256.of(modelPath)
            val manifest = CheckpointManifest(
                checkpointId = checkpointId,
                optimizerUpdates = progress.optimizerUpdates,
                tokensSeen = progress.tokensSeen,
                totalUpdates = totalUpdates,
                configSha256 = configSha256,
                modelSha256 = modelSha256,
            )
            writeJsonAtomically(checkpointDirectory.resolve(MANIFEST_FILE), manifest)
            writeTextAtomically(runDirectory.resolve(CHECKPOINTS_DIRECTORY).resolve(LATEST_FILE), "$checkpointId\n")
            return SavedCheckpoint(checkpointDirectory, manifest)
        } catch (exception: CheckpointException) {
            throw exception
        } catch (exception: IOException) {
            throw CheckpointException("Cannot save checkpoint $checkpointId: ${exception.message}", exception)
        }
    }

    fun loadLatest(
        runDirectory: Path,
        model: Block,
        manager: NDManager,
        expectedConfigSha256: String,
    ): LoadedCheckpoint {
        requireChecksum(expectedConfigSha256, "expectedConfigSha256")
        val checkpointsDirectory = runDirectory.resolve(CHECKPOINTS_DIRECTORY)
        val latestPath = checkpointsDirectory.resolve(LATEST_FILE)
        if (!Files.isRegularFile(latestPath)) throw CheckpointException("Latest checkpoint pointer does not exist: $latestPath")
        val checkpointId = try {
            Files.readString(latestPath).trim()
        } catch (exception: IOException) {
            throw CheckpointException("Cannot read $latestPath: ${exception.message}", exception)
        }
        if (!checkpointId.matches(CHECKPOINT_ID_PATTERN)) {
            throw CheckpointException("Invalid latest checkpoint ID: '$checkpointId'")
        }
        return load(checkpointsDirectory.resolve(checkpointId), model, manager, expectedConfigSha256)
    }

    fun load(
        checkpointDirectory: Path,
        model: Block,
        manager: NDManager,
        expectedConfigSha256: String,
    ): LoadedCheckpoint {
        requireChecksum(expectedConfigSha256, "expectedConfigSha256")
        val manifestPath = checkpointDirectory.resolve(MANIFEST_FILE)
        val modelPath = checkpointDirectory.resolve(MODEL_FILE)
        val manifest = readManifest(manifestPath)
        validateManifest(manifest, checkpointDirectory.fileName.toString(), expectedConfigSha256)
        if (!Files.isRegularFile(modelPath)) throw CheckpointException("Model parameters do not exist: $modelPath")
        val actualModelSha256 = Sha256.of(modelPath)
        if (actualModelSha256 != manifest.modelSha256) {
            throw CheckpointException(
                "Model checksum mismatch: expected ${manifest.modelSha256}, got $actualModelSha256",
            )
        }

        try {
            DataInputStream(BufferedInputStream(Files.newInputStream(modelPath))).use { input ->
                model.loadParameters(manager, input)
                if (input.read() != -1) throw CheckpointException("Model parameter file contains trailing data: $modelPath")
            }
            model.parameters.values().forEach { parameter ->
                parameter.array.setRequiresGradient(parameter.requiresGradient())
            }
        } catch (exception: MalformedModelException) {
            throw CheckpointException("Checkpoint parameters are incompatible with the model: ${exception.message}", exception)
        } catch (exception: IOException) {
            throw CheckpointException("Cannot load model parameters from $modelPath: ${exception.message}", exception)
        }
        return LoadedCheckpoint(
            checkpointDirectory,
            manifest,
            TrainingProgress(manifest.optimizerUpdates, manifest.tokensSeen),
        )
    }

    private fun readManifest(path: Path): CheckpointManifest {
        if (!Files.isRegularFile(path)) throw CheckpointException("Checkpoint manifest does not exist: $path")
        return try {
            Files.newBufferedReader(path).use { reader -> mapper.readValue(reader, CheckpointManifest::class.java) }
        } catch (exception: JsonProcessingException) {
            throw CheckpointException("Invalid checkpoint manifest $path: ${exception.originalMessage}", exception)
        } catch (exception: IOException) {
            throw CheckpointException("Cannot read checkpoint manifest $path: ${exception.message}", exception)
        }
    }

    private fun validateManifest(manifest: CheckpointManifest, directoryName: String, expectedConfigSha256: String) {
        val errors = buildList {
            if (manifest.schemaVersion != FORMAT_VERSION) add("schemaVersion must be $FORMAT_VERSION")
            if (manifest.checkpointId != directoryName || !manifest.checkpointId.matches(CHECKPOINT_ID_PATTERN)) {
                add("checkpointId must match its directory")
            }
            if (manifest.optimizerUpdates <= 0) add("optimizerUpdates must be positive")
            if (manifest.optimizerUpdates > MAX_CHECKPOINT_UPDATE) add("optimizerUpdates exceeds format 1 capacity")
            if (manifest.tokensSeen < 0) add("tokensSeen must be non-negative")
            if (manifest.totalUpdates < manifest.optimizerUpdates) add("totalUpdates must cover optimizerUpdates")
            if (manifest.configSha256 != expectedConfigSha256) add("configuration checksum does not match this run")
            if (manifest.modelFile != MODEL_FILE) add("modelFile must be '$MODEL_FILE'")
            if (!Sha256.isValid(manifest.modelSha256)) add("modelSha256 must be a SHA-256 value")
            if (!manifest.optimizerCounterRestored) add("optimizerCounterRestored must be true")
            if (manifest.optimizerMomentsRestored) add("optimizerMomentsRestored must remain false in format 1")
            if (!manifest.schedulerRestored) add("schedulerRestored must be true")
            if (manifest.randomStateRestored) add("randomStateRestored must remain false in format 1")
            if (manifest.exactTrainingResume) add("exactTrainingResume cannot be true without complete state")
        }
        if (errors.isNotEmpty()) throw CheckpointException("Incompatible checkpoint manifest: ${errors.joinToString("; ")}")
    }

    private fun validateProgress(progress: TrainingProgress, totalUpdates: Int) {
        if (progress.optimizerUpdates <= 0) throw CheckpointException("optimizerUpdates must be positive")
        if (progress.optimizerUpdates > MAX_CHECKPOINT_UPDATE) {
            throw CheckpointException("optimizerUpdates cannot exceed $MAX_CHECKPOINT_UPDATE in format $FORMAT_VERSION")
        }
        if (progress.tokensSeen < 0) throw CheckpointException("tokensSeen must be non-negative")
        if (totalUpdates < progress.optimizerUpdates) throw CheckpointException("totalUpdates must cover optimizerUpdates")
    }

    private fun requireChecksum(value: String, name: String) {
        if (!Sha256.isValid(value)) throw CheckpointException("$name must be 64 lowercase hexadecimal characters")
    }

    private fun writeJsonAtomically(path: Path, value: Any) = writeAtomically(path) { temporaryPath ->
        Files.newBufferedWriter(temporaryPath).use { writer -> mapper.writeValue(writer, value) }
    }

    private fun writeTextAtomically(path: Path, value: String) = writeAtomically(path) { temporaryPath ->
        Files.writeString(temporaryPath, value)
    }

    private fun writeAtomically(path: Path, writer: (Path) -> Unit) {
        path.parent?.let(Files::createDirectories)
        val temporaryPath = path.resolveSibling("${path.fileName}.tmp")
        try {
            Files.deleteIfExists(temporaryPath)
            writer(temporaryPath)
            try {
                Files.move(temporaryPath, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(temporaryPath, path, StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            Files.deleteIfExists(temporaryPath)
        }
    }

    companion object {
        const val FORMAT_VERSION = 1
        const val CHECKPOINTS_DIRECTORY = "checkpoints"
        const val MODEL_FILE = "model.params"
        const val MANIFEST_FILE = "manifest.json"
        const val LATEST_FILE = "latest.txt"
        const val MAX_CHECKPOINT_UPDATE = 99_999_999
        private val CHECKPOINT_ID_PATTERN = Regex("step-[0-9]{8}")

        fun checkpointId(update: Int): String = "step-%08d".format(update)

        private fun checkpointMapper(): ObjectMapper = ObjectMapper()
            .registerModule(KotlinModule.Builder().build())
            .enable(SerializationFeature.INDENT_OUTPUT)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(DeserializationFeature.FAIL_ON_NULL_FOR_PRIMITIVES)
    }
}

@JsonPropertyOrder(
    "schemaVersion",
    "checkpointId",
    "optimizerUpdates",
    "tokensSeen",
    "totalUpdates",
    "configSha256",
    "modelFile",
    "modelSha256",
    "optimizerCounterRestored",
    "optimizerMomentsRestored",
    "schedulerRestored",
    "randomStateRestored",
    "exactTrainingResume",
)
data class CheckpointManifest(
    val schemaVersion: Int = CheckpointStore.FORMAT_VERSION,
    val checkpointId: String,
    val optimizerUpdates: Int,
    val tokensSeen: Long,
    val totalUpdates: Int,
    val configSha256: String,
    val modelFile: String = CheckpointStore.MODEL_FILE,
    val modelSha256: String,
    val optimizerCounterRestored: Boolean = true,
    val optimizerMomentsRestored: Boolean = false,
    val schedulerRestored: Boolean = true,
    val randomStateRestored: Boolean = false,
    val exactTrainingResume: Boolean = false,
)

data class SavedCheckpoint(val directory: Path, val manifest: CheckpointManifest)

data class LoadedCheckpoint(
    val directory: Path,
    val manifest: CheckpointManifest,
    val progress: TrainingProgress,
)
