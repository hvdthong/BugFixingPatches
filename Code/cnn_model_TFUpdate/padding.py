def input_path_maxtext_maxcode_mapping(options):
    paths = []
    if options == "eq":
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    elif options == "extra":
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    elif options == "lbd":
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    else:
        print "You select wrong options"
        exit()
    return paths


def max_commit_message_codefile(path_file):
    commits = list(open(path_file, "r").readlines())
    msg_length = [len(c.strip().split("\t")[1].split()) for c in commits]
    return max(msg_length)


def max_commit_code(path_file):
    commits = list(open(path_file, "r").readlines())
    msg_length = []
    for c in commits:
        code = c.strip().split("\t")[1:]
        if len(code) > 0:
            msg_length.append(max([len(line.split()) for line in code]))
        else:
            msg_length.append(0)
        # print code
        # msg_length.append(max([len(line.split()) for line in code]))
        # exit()
        # for line in code:
        #     print line
        # exit()
    return max(msg_length)


def max_commit_message_all():
    eq_msg_path = input_path_maxtext_maxcode_mapping("eq")[0]
    extra_msg_path = input_path_maxtext_maxcode_mapping("extra")[0]
    lbd_msg_path = input_path_maxtext_maxcode_mapping("lbd")[0]
    print eq_msg_path, extra_msg_path, lbd_msg_path
    print max_commit_message_codefile(eq_msg_path)

    eq_addedcode_path = input_path_maxtext_maxcode_mapping("eq")[1]
    print max_commit_code(eq_addedcode_path)


max_commit_message_all()
