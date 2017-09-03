from process_data_3convs import padding_commit, build_vocab, build_input_data


pad_msg, pad_addedcode, pad_removedcode = padding_commit()
# build_vocab(sentences=pad_msg, options="msg")
vocab_code = build_vocab(sentences=pad_addedcode + pad_removedcode, options="code")
added_code = build_input_data(sentences=pad_addedcode, options="code", vocab=vocab_code, maxinput=100)
added_code = build_input_data(sentences=pad_removedcode, options="code", vocab=vocab_code, maxinput=100)