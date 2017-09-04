from process_data_3convs import read_input_datal, len_path_data


msg = read_input_data(options="msg")
addedcode = read_input_data(options="addedcode")
removedcode = read_input_data(options="removedcode")
print msg.shape, addedcode.shape, removedcode.shape
