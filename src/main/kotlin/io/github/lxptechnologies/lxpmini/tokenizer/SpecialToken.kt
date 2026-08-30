package io.github.lxptechnologies.lxpmini.tokenizer

enum class SpecialToken(
    val tokenText: String,
    val id: Int,
) {
    PAD("<pad>", 0),
    BOS("<bos>", 1),
    EOS("<eos>", 2),
    ;

    companion object {
        private val byId = entries.associateBy(SpecialToken::id)

        fun fromId(id: Int): SpecialToken? = byId[id]
    }
}
