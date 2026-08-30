package io.github.lxptechnologies.lxpmini.data

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test

class TokenPipelineTest {
    @Test
    fun `counts tokens through a bounded primitive buffer`() {
        val tokens = IntArray(20_001) { it }

        val count = CorpusTokenCounter.count(TokenReaderFactory { IntArrayTokenReader(tokens) }, bufferSize = 17)

        assertThat(count).isEqualTo(20_001)
    }

    @Test
    fun `range reader exposes only its disjoint interval`() {
        val destination = IntArray(10)

        val count = RangedTokenReader(IntArrayTokenReader(intArrayOf(10, 20, 30, 40, 50)), TokenRange(1, 4))
            .use { reader -> reader.read(destination) }

        assertThat(count).isEqualTo(3)
        assertThat(destination.copyOf(count)).containsExactly(20, 30, 40)
    }

    @Test
    fun `builds next-token inputs and targets with an exact one-token shift`() {
        val windowReader = SlidingWindowReader(
            tokenReader = IntArrayTokenReader(intArrayOf(10, 20, 30, 40, 50)),
            contextLength = 4,
        )

        val batch = TokenBatchReader(windowReader, requestedBatchSize = 1).use { it.readBatch() }

        assertThat(batch).isNotNull
        assertThat(batch!!.inputRow(0)).containsExactly(10, 20, 30, 40)
        assertThat(batch.targetRow(0)).containsExactly(20, 30, 40, 50)
        assertThat(batch.inputIds).hasSize(batch.batchSize * batch.sequenceLength)
    }

    @Test
    fun `slides across boundaries and drops only the final incomplete window`() {
        val reader = SlidingWindowReader(
            tokenReader = IntArrayTokenReader(IntArray(10) { it }),
            contextLength = 3,
            stride = 2,
        )
        val windows = readAllWindows(reader)
        val plan = WindowPlan(tokenCount = 10, contextLength = 3, stride = 2)

        assertThat(windows).containsExactly(
            intArrayOf(0, 1, 2, 3),
            intArrayOf(2, 3, 4, 5),
            intArrayOf(4, 5, 6, 7),
            intArrayOf(6, 7, 8, 9),
        )
        assertThat(plan.windowCount).isEqualTo(4)
        assertThat(plan.trailingTokenCount).isZero()
    }

    @Test
    fun `reports tokens in a final segment that cannot form a full window`() {
        val plan = WindowPlan(tokenCount = 9, contextLength = 4, stride = 4)
        val windows = readAllWindows(
            SlidingWindowReader(IntArrayTokenReader(IntArray(9) { it }), contextLength = 4, stride = 4),
        )

        assertThat(windows).containsExactly(
            intArrayOf(0, 1, 2, 3, 4),
            intArrayOf(4, 5, 6, 7, 8),
        )
        assertThat(plan.trailingTokenCount).isZero()

        val incompletePlan = WindowPlan(tokenCount = 8, contextLength = 4, stride = 4)
        assertThat(incompletePlan.windowCount).isEqualTo(1)
        assertThat(incompletePlan.trailingTokenCount).isEqualTo(3)
    }

    @Test
    fun `returns a smaller last batch unless drop-last is requested`() {
        val tokens = IntArray(13) { it }
        val keepReader = TokenBatchReader(
            SlidingWindowReader(IntArrayTokenReader(tokens), contextLength = 2, stride = 2),
            requestedBatchSize = 4,
        )
        val keepSizes = generateSequence(keepReader::readBatch).map(TokenBatch::batchSize).toList()
        keepReader.close()

        val dropReader = TokenBatchReader(
            SlidingWindowReader(IntArrayTokenReader(tokens), contextLength = 2, stride = 2),
            requestedBatchSize = 4,
            dropLastBatch = true,
        )
        val dropSizes = generateSequence(dropReader::readBatch).map(TokenBatch::batchSize).toList()
        dropReader.close()

        assertThat(keepSizes).containsExactly(4, 2)
        assertThat(dropSizes).containsExactly(4)
    }

    @Test
    fun `bounded shuffle is reproducible for a seed`() {
        val first = shuffledWindowStarts(seed = 42)
        val second = shuffledWindowStarts(seed = 42)
        val different = shuffledWindowStarts(seed = 43)

        assertThat(first).containsExactlyElementsOf(second)
        assertThat(first).isNotEqualTo(different)
        assertThat(first.sorted()).containsExactlyElementsOf((0..18 step 2).toList())
    }

    private fun shuffledWindowStarts(seed: Long): List<Int> {
        val delegate = SlidingWindowReader(
            IntArrayTokenReader(IntArray(21) { it }),
            contextLength = 2,
            stride = 2,
        )
        return BufferedShufflingWindowReader(delegate, bufferSize = 4, seed = seed).use { reader ->
            readAllWindows(reader).map { window -> window.first() }
        }
    }

    private fun readAllWindows(reader: WindowReader): List<IntArray> {
        val result = mutableListOf<IntArray>()
        reader.use {
            val destination = IntArray(reader.contextLength + 1)
            while (reader.readWindow(destination)) result += destination.copyOf()
        }
        return result
    }
}
