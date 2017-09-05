from process_data_3convs import read_input_data, read_input_codefile_all
from data_loading_3convs import model_parameters, loading_training_data

maxtext, maxcode, maxline = 175, 250, 20
msg = read_input_data(options="msg", maxtext=maxtext, maxcode=maxcode, maxline=maxline)
addedcode = read_input_data(options="addedcode")
removedcode = read_input_data(options="removedcode")
codefile = read_input_codefile_all()
print msg.shape, addedcode.shape, removedcode.shape, codefile.shape

tf = model_parameters()
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

cnn_model = "three_convs"
loading_training_data(msg=msg, addedcode=addedcode, removedcode=removedcode, codefile=codefile)


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)


