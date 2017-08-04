def check_text_code(path_file):
    commits = list(open(path_file, "r").readlines())
    flag = [c for c in commits if len(c.split("\t")) == 2]
    if len(flag) == len(commits):
        print "Congrats! Your file is correct"
    else:
        print "Please recheck your file, something wrongs"


path_eq_neq = "./data/eq100_line_aug1.neg.contain"
path_eq_pos = "./data/eq100_line_aug1.pos.contain"
path_extra = "./data/extra100_line_aug1.neg.contain"
path_lbl_neg = "./data/lbd100_line_aug1.neg.contain"
path_lbl_pos = "./data/lbd100_line_aug1.pos.contain"
check_text_code(path_file=path_eq_neq)
check_text_code(path_file=path_eq_pos)
check_text_code(path_file=path_eq_neq)
check_text_code(path_file=path_lbl_neg)
check_text_code(path_file=path_lbl_pos)