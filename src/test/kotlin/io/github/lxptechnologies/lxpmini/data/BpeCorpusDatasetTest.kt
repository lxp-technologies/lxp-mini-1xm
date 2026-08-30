package io.github.lxptechnologies.lxpmini.data

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerTrainer
import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Files
import java.nio.file.Path

class BpeCorpusDatasetTest {
    @TempDir
    lateinit var temporaryDirectory: Path

    @Test
    fun `train and validation readers stay inside their disjoint token ranges`() {
        val trainCorpus = temporaryDirectory.resolve("train.txt")
        val validationCorpus = temporaryDirectory.resolve("validation.txt")
        Files.writeString(trainCorpus, "abcdefghijklmnopqrstuvwx")
        Files.writeString(validationCorpus, "0123456789")
        val tokenizer = BpeTokenizerTrainer().train(Files.readAllBytes(trainCorpus), 259).tokenizer
        val dataset = BpeCorpusDataset(
            trainCorpus,
            validationCorpus,
            tokenizer,
            contextLength = 4,
            batchSize = 8,
        )

        val trainTargets = dataset.trainBatches(0, 42).use { reader ->
            reader.asSequence().flatMap { it.targetIds.asSequence() }.toList()
        }
        val validationInputs = dataset.validationBatches().use { reader ->
            reader.asSequence().flatMap { it.inputIds.asSequence() }.toList()
        }

        assertThat(trainTargets).doesNotContainAnyElementsOf(validationInputs)
        assertThat(dataset.trainTokenCount).isEqualTo(24)
        assertThat(dataset.validationTokenCount).isEqualTo(10)
        assertThat(dataset.trainPlan.windowCount).isPositive()
        assertThat(dataset.validationPlan.windowCount).isPositive()
    }
}
