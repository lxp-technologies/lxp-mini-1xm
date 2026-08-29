package io.github.lxptechnologies.lxpmini

import io.github.lxptechnologies.lxpmini.cli.LxpMiniCommand
import picocli.CommandLine
import kotlin.system.exitProcess

fun main(args: Array<String>) {
    exitProcess(CommandLine(LxpMiniCommand()).execute(*args))
}
