package io.github.lxptechnologies.lxpmini.tokenizer

data class TokenPair(
    val leftId: Int,
    val rightId: Int,
) : Comparable<TokenPair> {
    override fun compareTo(other: TokenPair): Int {
        val leftComparison = leftId.compareTo(other.leftId)
        return if (leftComparison != 0) leftComparison else rightId.compareTo(other.rightId)
    }
}

data class BpeMerge(
    val pair: TokenPair,
    val resultId: Int,
    val trainingFrequency: Int,
)
