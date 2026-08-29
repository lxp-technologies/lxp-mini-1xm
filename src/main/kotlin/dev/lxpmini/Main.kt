package dev.lxpmini

import dev.lxpmini.cli.LxpMiniCommand
import picocli.CommandLine
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    exitProcess(CommandLine(LxpMiniCommand()).execute(*args))
}
