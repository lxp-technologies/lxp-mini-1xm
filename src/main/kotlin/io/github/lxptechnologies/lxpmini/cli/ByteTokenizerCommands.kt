package io.github.lxptechnologies.lxpmini.cli

import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizer
import io.github.lxptechnologies.lxpmini.tokenizer.ByteTokenizerArtifactStore
import io.github.lxptechnologies.lxpmini.tokenizer.SpecialToken
import io.github.lxptechnologies.lxpmini.tokenizer.TokenizerException
import picocli.CommandLine.ArgGroup
import picocli.CommandLine.Command
import picocli.CommandLine.Option
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.Callable

@Command(
    name = "tokenizer",
    description = ["Inspect and create tokenizers."],
    subcommands = [ByteTokenizerCommand::class],
)
class TokenizerCommand : Runnable {
    override fun run() {
        println("Choose a tokenizer command. Try: lxp-mini tokenizer byte --help")
    }
}

@Command(
    name = "byte",
    description = ["Explore the UTF-8 byte tokenizer introduced in PR02."],
    subcommands = [ByteTokenizerInspectCommand::class, ByteTokenizerCreateCommand::class],
)
class ByteTokenizerCommand : Runnable {
    override fun run() {
        println("Choose a byte tokenizer command. Try: lxp-mini tokenizer byte inspect --help")
    }
}

@Command(
    name = "inspect",
    mixinStandardHelpOptions = true,
    description = ["Show how UTF-8 text maps to bytes and token IDs."],
)
class ByteTokenizerInspectCommand(
    private val tokenizer: ByteTokenizer = ByteTokenizer(),
) : Callable<Int> {
    @ArgGroup(
        exclusive = true,
        multiplicity = "1",
        heading = "Input (choose one):%n",
    )
    lateinit var input: ByteTokenizerTextInput

    @Option(names = ["--add-bos"], description = ["Add the beginning-of-sequence token."])
    var addBos: Boolean = false

    @Option(names = ["--add-eos"], description = ["Add the end-of-sequence token."])
    var addEos: Boolean = false

    override fun call(): Int = try {
        val inputText = input.readText()
        val utf8Bytes = inputText.toByteArray(StandardCharsets.UTF_8)
        val tokenIds = tokenizer.encode(inputText, addBos = addBos, addEos = addEos)
        val decodedText = tokenizer.decode(tokenIds)

        println("Text:               \"${inputText.visibleWhitespace()}\"")
        println("UTF-8 bytes:        ${utf8Bytes.toUnsignedDisplay()}")
        println("Token IDs:          ${tokenIds.contentToString()}")
        println("Decoded text:       \"${decodedText.visibleWhitespace()}\"")
        println("Vocabulary size:    ${tokenizer.vocabularySize}")
        println("Byte token formula: tokenId = unsignedByte + ${ByteTokenizer.BYTE_TOKEN_OFFSET}")
        println(
            "Special tokens:     ${SpecialToken.PAD.tokenText}=${SpecialToken.PAD.id}, " +
                "${SpecialToken.BOS.tokenText}=${SpecialToken.BOS.id}, " +
                "${SpecialToken.EOS.tokenText}=${SpecialToken.EOS.id}",
        )
        0
    } catch (exception: IOException) {
        System.err.println("Tokenizer input error: ${exception.message}")
        2
    }
}

class ByteTokenizerTextInput {
    @Option(names = ["--text"], description = ["Text supplied directly on the command line."])
    var text: String? = null

    @Option(
        names = ["--text-file"],
        paramLabel = "<file>",
        description = ["UTF-8 text file; recommended for Unicode experiments on Windows."],
    )
    var textFile: Path? = null

    fun readText(): String = text ?: Files.readString(requireNotNull(textFile), StandardCharsets.UTF_8)
}

@Command(
    name = "create",
    mixinStandardHelpOptions = true,
    description = ["Write the versioned byte tokenizer artifact."],
)
class ByteTokenizerCreateCommand(
    private val tokenizer: ByteTokenizer = ByteTokenizer(),
    private val artifactStore: ByteTokenizerArtifactStore = ByteTokenizerArtifactStore(),
) : Callable<Int> {
    @Option(
        names = ["--output"],
        required = true,
        paramLabel = "<file>",
        description = ["Destination tokenizer.json path."],
    )
    lateinit var outputPath: Path

    override fun call(): Int = try {
        artifactStore.save(tokenizer, outputPath)
        println("Byte tokenizer written to $outputPath")
        0
    } catch (exception: TokenizerException) {
        System.err.println("Tokenizer error: ${exception.message}")
        2
    }
}

private fun ByteArray.toUnsignedDisplay(): String = joinToString(prefix = "[", postfix = "]") {
    (it.toInt() and 0xFF).toString()
}

private fun String.visibleWhitespace(): String = buildString {
    this@visibleWhitespace.forEach { character ->
        append(
            when (character) {
                '\n' -> "\\n"
                '\r' -> "\\r"
                '\t' -> "\\t"
                else -> character
            },
        )
    }
}
