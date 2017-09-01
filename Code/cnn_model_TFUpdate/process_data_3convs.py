from padding import max_commit_file, padding_file


def input_stable_path(options):
    paths = []
    if options == "eq":
        paths.append("./eq100_line_aug1.out.stable.maxtext175.maxcode250")
    elif options == "extra":
        paths.append("./extra100_line_aug1.out.stable.maxtext175.maxcode250")
    elif options == "lbd":
        paths.append("./lbd100_line_aug1.out.stable.maxtext175.maxcode250")
    else:
        print "Wrong options"
        exit()
    return paths


def finding_max_commit():
    maxlen_msg_eq, maxlen_code_eq, maxline_code_eq, maxlen_file_eq = max_commit_file(options="eq")
    maxlen_msg_extra, maxlen_code_extra, maxline_code_extra, maxlen_file_extra = max_commit_file(options="extra")
    maxlen_msg_lbd, maxlen_code_lbd, maxline_code_lbd, maxlen_file_lbd = max_commit_file(options="lbd")

    return max(maxlen_msg_eq, maxlen_msg_extra, maxlen_msg_lbd), \
           max(maxlen_code_eq, maxlen_code_extra, maxlen_code_lbd), \
           max(maxline_code_eq, maxline_code_extra, maxline_code_lbd), \
           max(maxlen_file_eq, maxlen_file_extra, maxlen_file_lbd)


def get_padding(pad):
    return [p.split("\t")[1:]for p in pad]


def padding_commit():
    maxlen_msg, maxlen_commitcode, maxline_commitcode, maxlen_file = finding_max_commit()
    print maxlen_msg, maxlen_commitcode, maxline_commitcode, maxlen_file

    pad_msg_eq, pad_addedcode_eq, pad_removedcode_eq = padding_file(options="eq", maxlen_msg=maxlen_msg,
                                                                    maxlen_code=maxlen_commitcode,
                                                                    maxline_code=maxline_commitcode)
    pad_msg_extra, pad_addedcode_extra, pad_removedcode_extra = padding_file(options="extra", maxlen_msg=maxlen_msg,
                                                                             maxlen_code=maxlen_commitcode,
                                                                             maxline_code=maxline_commitcode)
    pad_msg_lbd, pad_addedcode_lbd, pad_removedcode_lbd = padding_file(options="lbd", maxlen_msg=maxlen_msg,
                                                                       maxlen_code=maxlen_commitcode,
                                                                       maxline_code=maxline_commitcode)
    print len(pad_msg_eq), len(pad_msg_extra), len(pad_msg_lbd)
    pad_msg = pad_msg_eq + pad_msg_extra + pad_addedcode_lbd
    pad_addedcode = pad_addedcode_eq + pad_addedcode_extra + pad_addedcode_lbd
    pad_removedcode = pad_removedcode_eq + pad_removedcode_extra + pad_removedcode_lbd
    pad_msg, pad_addedcode, pad_removedcode = get_padding(pad_msg), \
                                              get_padding(pad_addedcode), get_padding(pad_removedcode)
    print len(pad_msg), len(pad_addedcode), len(pad_removedcode)
    return pad_msg, pad_addedcode, pad_removedcode
