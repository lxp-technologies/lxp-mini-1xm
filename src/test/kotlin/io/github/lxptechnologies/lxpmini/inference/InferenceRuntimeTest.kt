package io.github.lxptechnologies.lxpmini.inference

import ai.djl.Device
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import io.github.lxptechnologies.lxpmini.checkpoint.CheckpointStore
import io.github.lxptechnologies.lxpmini.checkpoint.RunStore
import io.github.lxptechnologies.lxpmini.checkpoint.Sha256
import io.github.lxptechnologies.lxpmini.cli.CheckpointDemoCommand
import io.github.lxptechnologies.lxpmini.config.ConfigLoader
import io.github.lxptechnologies.lxpmini.generation.AutoregressiveGenerator
import io.github.lxptechnologies.lxpmini.generation.GenerationResult
import io.github.lxptechnologies.lxpmini.generation.SamplingOptions
import io.github.lxptechnologies.lxpmini.generation.SamplingStrategy
import io.github.lxptechnologies.lxpmini.generation.TokenSampler
import io.github.lxptechnologies.lxpmini.model.DecoderLanguageModel
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import org.assertj.core.api.Assertions.assertThat
import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.junit.jupiter.api.parallel.ResourceAccessMode
import org.junit.jupiter.api.parallel.ResourceLock
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Executors

@ResourceLock(value = "DJL_ENGINE", mode = ResourceAccessMode.READ_WRITE)
class InferenceRuntimeTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `runtime loads once matches PR11 generation and releases every request`() {
        val artifacts = createArtifacts()
        val request = TokenGenerationRequest(
            promptTokenIds = ByteTokenizer().encode("abc"),
            maxNewTokens = 3,
            sampling = SamplingOptions(strategy = SamplingStrategy.GREEDY),
        )
        val expected = generateWithPr11Lifecycle(artifacts.runDirectory, request)
        val runtime = InferenceRuntimeLoader().load(MODEL_ID, artifacts.runDirectory, artifacts.tokenizerPath)

        assertThat(runtime.metadata.modelId).isEqualTo(MODEL_ID)
        assertThat(runtime.metadata.kind).isEqualTo(InferenceModelKind.BASE)
        assertThat(runtime.metadata.concurrencyPolicy).isEqualTo(InferenceConcurrencyPolicy.SERIALIZED)
        assertThat(runtime.generate(request).generatedTokenIds).containsExactly(*expected.generatedTokenIds)

        val before = runtime.diagnostics()
        repeat(100) {
            assertThat(runtime.generate(request).generatedTokenIds).containsExactly(*expected.generatedTokenIds)
        }
        val afterRepeatedRequests = runtime.diagnostics()
        assertThat(afterRepeatedRequests.completedRequests).isEqualTo(101)
        assertThat(afterRepeatedRequests.managedArrayCount).isEqualTo(before.managedArrayCount)

        val executor = Executors.newFixedThreadPool(4)
        try {
            val futures = List(12) { executor.submit<IntArray> { runtime.generate(request).generatedTokenIds } }
            futures.forEach { future -> assertThat(future.get()).containsExactly(*expected.generatedTokenIds) }
        } finally {
            executor.shutdownNow()
        }
        assertThat(runtime.diagnostics().completedRequests).isEqualTo(113)
        assertThat(runtime.diagnostics().managedArrayCount).isEqualTo(before.managedArrayCount)

        Files.delete(artifacts.tokenizerPath)
        Files.delete(artifacts.runDirectory.resolve(RunStore.CONFIG_FILE))
        Files.delete(artifacts.runDirectory.resolve(CheckpointStore.CHECKPOINTS_DIRECTORY).resolve("latest.txt"))
        assertThat(runtime.generate(request).generatedTokenIds).containsExactly(*expected.generatedTokenIds)

        runtime.close()
        runtime.close()
        assertThat(runtime.isClosed).isTrue()
        assertThatThrownBy { runtime.generate(request) }
            .isInstanceOf(InferenceException::class.java)
            .hasMessageContaining("is closed")
    }

    @Test
    fun `loader rejects an unstable external model identifier`() {
        val artifacts = createArtifacts()

        assertThatThrownBy {
            InferenceRuntimeLoader().load("LXP Mini/latest", artifacts.runDirectory, artifacts.tokenizerPath)
        }.isInstanceOf(InferenceException::class.java)
            .hasMessageContaining("modelId must match")
    }

    private fun createArtifacts(): InferenceArtifacts {
        val runDirectory = temporaryDirectory.resolve("run-${System.nanoTime()}")
        val tokenizerPath = temporaryDirectory.resolve("tokenizer-${System.nanoTime()}.json")
        val command = CheckpointDemoCommand().apply {
            configPath = Path.of("configs/lab-pr09-tiny.yaml").toAbsolutePath()
            this.runDirectory = runDirectory
            beforeUpdates = 5
            afterUpdates = 1
        }
        check(command.call() == 0)
        ByteTokenizerArtifactStore().save(ByteTokenizer(), tokenizerPath)
        return InferenceArtifacts(runDirectory, tokenizerPath)
    }

    private fun generateWithPr11Lifecycle(
        runDirectory: Path,
        request: TokenGenerationRequest,
    ): GenerationResult {
        val configPath = runDirectory.resolve(RunStore.CONFIG_FILE)
        val config = ConfigLoader().load(configPath)
        return NDManager.newBaseManager(Device.cpu()).use { manager ->
            DecoderLanguageModel(manager, config.model).use { model ->
                CheckpointStore().loadLatest(runDirectory, model, manager, Sha256.of(configPath))
                val parameterStore = ParameterStore(manager, false)
                val generator = AutoregressiveGenerator(
                    config.model.contextLength,
                    config.model.vocabSize,
                    TokenSampler(request.seed),
                ) { context ->
                    manager.newSubManager().use { temporary ->
                        val input = temporary.create(
                            context.map(Int::toLong).toLongArray(),
                            Shape(1, context.size.toLong()),
                        )
                        val logits = model.forward(parameterStore, NDList(input), false).singletonOrThrow()
                        logits.get("0, ${context.lastIndex}, :").toFloatArray()
                    }
                }
                generator.generate(
                    request.promptTokenIds,
                    request.maxNewTokens,
                    SpecialToken.EOS.id,
                    request.sampling,
                )
            }
        }
    }

    private data class InferenceArtifacts(val runDirectory: Path, val tokenizerPath: Path)

    private companion object {
        const val MODEL_ID = "lxp-mini-pr14-test"
    }
}
