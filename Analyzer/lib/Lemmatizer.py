from germalemma import GermaLemma


def lemmatize(tokens):
    lemmatizer = GermaLemma()
    new_tokens = []
    for token in tokens:
        lemmata_pos = []
        for i, t in enumerate(token):
            try:
                l = lemmatizer.find_lemma(t[0], t[1])
            except ValueError:
                l = t[0]
            lemmata_pos.append((l, t[1]))
        new_tokens.append(lemmata_pos)

    return new_tokens
