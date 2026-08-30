package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.BpeTokenizerTrainer
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import picocli.CommandLine.ArgGroup
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.Locale
import java.util.concurrent.Callable

@Command(
    name = "bpe",
    description = ["Train and inspect the deterministic byte-level BPE tokenizer introduced in PR03."],
    subcommands = [BpeTokenizerTrainCommand::class, BpeTokenizerInspectCommand::class],
)
class BpeTokenizerCommand : Runnable {
    override fun run() {
        println("Choose a BPE tokenizer command. Try: lxp-mini tokenizer bpe train --help")
    }
}

@Command(
    name = "train",
    mixinStandardHelpOptions = true,
    description = ["Train byte-level BPE merges from a corpus and write tokenizer.json."],
)
class BpeTokenizerTrainCommand(
    private val trainer: BpeTokenizerTrainer = BpeTokenizerTrainer(),
    private val artifactStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
) : Callable<Int> {
    @Option(names = ["--input"], required = true, paramLabel = "<file>", description = ["Training corpus read as raw bytes."])
    lateinit var inputPath: Path

    @Option(names = ["--vocab-size"], required = true, paramLabel = "<count>", description = ["Target size, including 3 special and 256 byte tokens."])
    var vocabularySize: Int = 0

    @Option(names = ["--output"], required = true, paramLabel = "<file>", description = ["Destination tokenizer.json path."])
    lateinit var outputPath: Path

    override fun call(): Int = try {
        val corpus = Files.readAllBytes(inputPath)
        val trained = trainer.train(corpus, vocabularySize)
        artifactStore.save(trained, outputPath)

        println("BPE tokenizer written to $outputPath")
        println("Corpus bytes:       ${trained.metadata.corpusByteCount}")
        println("Corpus SHA-256:     ${trained.metadata.corpusSha256}")
        println("Vocabulary size:    ${trained.tokenizer.vocabularySize}")
        println("Learned merges:     ${trained.tokenizer.merges.size}")
        0
    } catch (exception: IOException) {
        System.err.println("Tokenizer input error: ${exception.message}")
        2
    } catch (exception: TokenizerException) {
        System.err.println("Tokenizer error: ${exception.message}")
        2
    }
}

@Command(
    name = "inspect",
    mixinStandardHelpOptions = true,
    description = ["Encode text with a trained BPE artifact and display the learned pieces."],
)
class BpeTokenizerInspectCommand(
    private val artifactStore: BpeTokenizerArtifactStore = BpeTokenizerArtifactStore(),
) : Callable<Int> {
    @Option(names = ["--tokenizer"], required = true, paramLabel = "<file>", description = ["Trained tokenizer.json path."])
    lateinit var tokenizerPath: Path

    @ArgGroup(exclusive = true, multiplicity = "1", heading = "Input (choose one):%n")
    lateinit var input: TokenizerTextInput

    @Option(names = ["--add-bos"], description = ["Add the beginning-of-sequence token."])
    var addBos: Boolean = false

    @Option(names = ["--add-eos"], description = ["Add the end-of-sequence token."])
    var addEos: Boolean = false

    @Option(names = ["--show-merges"], defaultValue = "10", paramLabel = "<count>", description = ["Number of learned merges to display."])
    var showMerges: Int = 10

    @Option(names = ["--show-vocabulary"], defaultValue = "10", paramLabel = "<count>", description = ["Number of learned vocabulary entries to display."])
    var showVocabulary: Int = 10

    @Option(names = ["--show-pieces"], defaultValue = "20", paramLabel = "<count>", description = ["Maximum encoded pieces to display."])
    var showPieces: Int = 20

    @Option(names = ["--summary-only"], description = ["Hide complete text, byte, token ID and piece listings."])
    var summaryOnly: Boolean = false

    override fun call(): Int = try {
        requireNonNegativeLimits()
        val trained = artifactStore.load(tokenizerPath)
        val tokenizer = trained.tokenizer
        val inputText = input.readText()
        val utf8Bytes = inputText.toByteArray(StandardCharsets.UTF_8)
        val tokenIds = tokenizer.encode(inputText, addBos = addBos, addEos = addEos)
        val decodedText = tokenizer.decode(tokenIds)

        if (!summaryOnly) {
            println("Text:               \"${inputText.visibleWhitespace()}\"")
            println("UTF-8 bytes:        ${utf8Bytes.toUnsignedDisplay()}")
            println("Token IDs:          ${tokenIds.contentToString()}")
            println("Decoded text:       \"${decodedText.visibleWhitespace()}\"")
        }
        println("Byte count:         ${utf8Bytes.size}")
        println("Token count:        ${tokenIds.size}")
        val bytesPerToken = if (tokenIds.isEmpty()) 0.0 else utf8Bytes.size.toDouble() / tokenIds.size
        println("Bytes per token:    ${"%.3f".format(Locale.ROOT, bytesPerToken)}")
        println("Round-trip exact:   ${decodedText == inputText}")
        println("Vocabulary size:    ${tokenizer.vocabularySize}")
        println("Corpus SHA-256:     ${trained.metadata.corpusSha256}")
        if (!summaryOnly) {
            println("Pieces:")
            tokenIds.take(showPieces).forEach { tokenId -> println("  ${tokenizer.describeToken(tokenId)}") }
            if (tokenIds.size > showPieces) {
                println("  ... ${tokenIds.size - showPieces} additional pieces hidden")
            }
        }

        println("First learned vocabulary entries:")
        (ByteTokenizer.VOCABULARY_SIZE until tokenizer.vocabularySize)
            .take(showVocabulary)
            .forEach { tokenId -> println("  ${tokenizer.describeToken(tokenId)}") }

        println("First learned merges:")
        tokenizer.merges.take(showMerges).forEachIndexed { index, merge ->
            println(
                "  #${index + 1}: (${merge.pair.leftId}, ${merge.pair.rightId}) -> " +
                    "${merge.resultId}, training frequency=${merge.trainingFrequency}",
            )
        }
        0
    } catch (exception: IOException) {
        System.err.println("Tokenizer input error: ${exception.message}")
        2
    } catch (exception: TokenizerException) {
        System.err.println("Tokenizer error: ${exception.message}")
        2
    }

    private fun requireNonNegativeLimits() {
        if (showMerges < 0 || showVocabulary < 0 || showPieces < 0) {
            throw TokenizerException("--show-merges, --show-vocabulary and --show-pieces must be non-negative")
        }
    }
}

private fun BpeTokenizer.describeToken(tokenId: Int): String {
    val specialToken = SpecialToken.fromId(tokenId)
    if (specialToken != null) return "$tokenId = ${specialToken.tokenText}"

    val bytes = bytesForToken(tokenId)
    val text = String(bytes, StandardCharsets.UTF_8).visibleWhitespace()
    return "$tokenId = bytes ${bytes.toUnsignedDisplay()}, UTF-8 view \"$text\""
}
