import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import multi_gpu_model

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
import generator
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]="3"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 64
TARGET_SIZE=(224, 224)
EPOCHS = 100
NO_GPU = 2
lr = 1e-3

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocessing_function

#Shape_Texture: (198,) -3.07:3.03, Expr: (29,) -1.28:0.68, Pose: (6,) -1.51:2542.07
def get_pretrained_model():
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = MobileNetV2(weights='imagenet', include_top=False)
    # base_model = VGG16(weights='imagenet', include_top=False)
    # base_model = ResNet50(weights='imagenet', include_top=False)
    # base_model = DenseNet121(weights='imagenet', include_top=False)
    # base_model = VGG19(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    Shape_Texture = Dense(198, activation='linear', name="Shape_Texture")(x)
    Expr = Dense(29, activation='linear', name="Expr")(x)
    Pose = Dense(6, activation='linear', name="Pose")(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=[Shape_Texture, Expr, Pose])
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers

    # for layer in base_model.layers:
    #     layer.trainable = False
    return model

# model = get_pretrained_model()
try:
    model = multi_gpu_model(get_pretrained_model(), gpus=NO_GPU)
except Exception as e:
    print(e)
    model = get_pretrained_model()
model.summary()

losses = {
	"Shape_Texture": "mse",
	"Expr": "mse",
	"Pose": "mse",
}

# opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=opt, loss=losses, metrics=['mse'])
filepath="./models/MobileNetV2-{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=1/np.sqrt(10),
                              patience=3, min_lr=0)
callbacks = [checkpointer, tensorboard_callback, reduce_lr]
# model.load_weights('models/VGG16_weights-improvement-49-0.47-0.81.hdf5')


# Quantization aware training
# sess = tf.keras.backend.get_session()
# tf.contrib.quantize.create_training_graph(sess.graph)
# sess.run(tf.global_variables_initializer())

# You can plot the quantize training graph on tensorboard
# tf.summary.FileWriter('/workspace/tensorboard', graph=sess.graph)

from imutils import paths
import numpy as np
import glob

train_path = 'data/new_png/preprocessed_img_train'
# test_path = 'data/new_png/preprocessed_img_test'
train_label_path = 'data/new_png/label_train_norm'
# test_label_path = 'data/new_png/label_test_norm'

# TODO: ok for training set, failed for test set.
def get_data_paths(label_path, image_path):
    image_paths = []
    label_paths = []
    label_paths_temp = glob.glob(label_path+'/**/*.dict')
    for p in label_paths_temp:
        p_i = image_path + '/' + p[-20:-4]+'jpg'
        if p[-20]!='n' or os.path.exists(p_i) is False:
            if p[-20]!='n':
                print(p)
            else:
                print(p_i)
            # os.remove(p)
            continue
        label_paths.append(p)
        image_paths.append(p_i)

    return image_paths, label_paths

def load_pb_model(model_path):
    f = gfile.FastGFile(model_path, 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess = K.get_session()
    sess.graph.as_default()
    # Import a serialized TensorFlow `GraphDef` protocol buffer
    # and place into the current default `Graph`.
    tf.import_graph_def(graph_def)
    return sess

# For 1 input
def pb_model_predict(sess, X, input_tensor_name, output_tensor_names):
    output_tensor = None
    if type(output_tensor_names) is list:
        output_tensor=[]
        for name in output_tensor_names:
            output_tensor.append(sess.graph.get_tensor_by_name(name))
    else:
        output_tensor = sess.graph.get_tensor_by_name(output_tensor_names)
    tensorflow_predictions = sess.run(output_tensor, {input_tensor_name: X})
    return tensorflow_predictions

def inspect_model_main():
    import sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir) 
    from main_inference import *
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    face_detection_model = getFaceDetectionModel()
    img_path = './images/Disgust_71_1.jpg'
    # img_path = './DistillModel/data/new/raw/n000001/0001_01.jpg'
    # label_dict=None
    # with open('./DistillModel/data/new/norm_label_train/n000001/0001_01.dict', 'rb') as dictionary_file:
    #     label_dict = pickle.load(dictionary_file)
    # temp_img = imread('./DistillModel/data/new/preprocessed_img_train/n000001/0001_01.jpg')

    img = cv2.imread(img_path)
    assert img is not None, "Loaded image failed"
    tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

    with tf.device(dev):
        Shape_Texture, Expr, Pose, faces, model = getFaceParams(img, face_detection_model)
    #     base_S = utils.get_base_S(model)
        dtype='>f8'
    #     utils.array_to_binary(base_S, './base_S.bin',dtype=dtype)
    #     utils.array_to_binary(model.expEV, './model_expEV.bin',dtype=dtype)
    #     utils.array_to_binary(model.expPC, './model_expPC.bin',dtype=dtype)
    #     utils.array_to_binary(model.shapeEV, './model_shapeEV.bin',dtype=dtype)
    #     utils.array_to_binary(model.shapePC, './model_shapePC.bin',dtype=dtype)

        name = "Disgust"
    #     utils.array_to_binary(Pose, './'+name+'_Pose.bin',dtype=dtype)
    #     utils.array_to_binary(Expr, './'+name+'_Expr.bin',dtype=dtype)
    #     utils.array_to_binary(Shape_Texture, './'+name+'_Shape_Texture.bin',dtype=dtype)

        with open('./base_S.bin', 'rb') as f:
            base_S_load = np.fromfile(f, dtype=dtype)
        with open('./model_expEV.bin', 'rb') as f:
            model_expEV = np.fromfile(f, dtype=dtype)
        with open('./model_expPC.bin', 'rb') as f:
            model_expPC = np.fromfile(f, dtype=dtype)
            model_expPC = np.reshape(model_expPC,(-1,29))
        with open('./model_shapeEV.bin', 'rb') as f:
            model_shapeEV = np.fromfile(f, dtype=dtype)
        with open('./model_shapePC.bin', 'rb') as f:
            model_shapePC = np.fromfile(f, dtype=dtype)
            model_shapePC = np.reshape(model_shapePC,(-1,99))

    #     with open('./'+name+'_Pose.bin', 'rb') as f:
    #         Pose2 = np.fromfile(f, dtype=dtype)
    #     with open('./'+name+'_Expr.bin', 'rb') as f:
    #         Expr2 = np.fromfile(f, dtype=dtype)
        with open('./'+name+'_Shape_Texture.bin', 'rb') as f:
            Shape_Texture2 = np.fromfile(f, dtype=dtype)

        bbox_dict = getFaceBBox(img, face_detection_model)
        preprocessed_img = preProcessImage(img, bbox_dict)
        cv2.imwrite('./'+name+'_preprocessed.png',preprocessed_img,[cv2.IMWRITE_PNG_COMPRESSION, 0]) # compressed when saved ?

        rgb_img = cv2.cvtColor(preprocessed_img,cv2.COLOR_BGR2RGB) # different than read image with skimage.io.imread
        imagenet_preprocessed_img = preprocessing_function(rgb_img_same_size)
        reshape_preprocessed_img = np.array([imagenet_preprocessed_img])

        sess_expr = load_pb_model("./DistillModel/models/Expr_MobileNetV2-98-0.08.pb")
        Expr2 = pb_model_predict(sess_expr, reshape_preprocessed_img, input_tensor_name='import/input_1:0',
            output_tensor_names='import/Expr/BiasAdd:0')
        Expr2=Expr2[0]
        sess_expr.close()

        K.clear_session()

        sess_pose = load_pb_model("./DistillModel/models/Pose_MobileNetV2-97-0.01.pb")
        Pose2 = pb_model_predict(sess_pose, reshape_preprocessed_img, input_tensor_name='import/input_1:0',
            output_tensor_names='import/Pose/BiasAdd:0')
        Pose2=Pose2[0]
        sess_pose.close()

        with open('./Expr_mean_test.bin', 'rb') as f:
            Expr2_mean_load = np.fromfile(f, dtype=dtype)
        with open('./Expr_var_test.bin', 'rb') as f:
            Expr2_var_load = np.fromfile(f, dtype=dtype)
        with open('./Pose_mean_test.bin', 'rb') as f:
            Pose2_mean_load = np.fromfile(f, dtype=dtype)
        with open('./Pose_var_test.bin', 'rb') as f:
            Pose2_var_load = np.fromfile(f, dtype=dtype)
        Pose2=Pose2*(Pose2_var_load**0.5)+Pose2_mean_load
        Expr2=Expr2*(Expr2_var_load**0.5)+Expr2_mean_load


        SEP,TEP,d0 = utils.projectBackBFM_withEP(copy.deepcopy(model), copy.deepcopy(Shape_Texture), copy.deepcopy(Expr), copy.deepcopy(Pose))
        SEP2,d1 = utils.update_pose_expr(model_expEV, model_expPC, model_shapeEV, model_shapePC, base_S_load, Expr2, Pose2, Shape_Texture2)
        
    #     #TODO save and load make array different from original.
    #     # print(np.array_equal(d0['S_RT'], d1['S_RT']))
    #     # print('====================')
    #     # print(np.array_equal(d0['E'], d1['E']))
 
        utils.write_ply(mesh_folder + '/'+name+'_original.ply', SEP, TEP, faces)
        utils.write_ply(mesh_folder + '/'+name+'_distill.ply', SEP2, TEP, faces)

# image_paths_test,label_paths_test = get_data_paths('/home/thangdn3/Desktop/Expression-Net/DistillModel/'+test_label_path, test_path)
image_paths,label_paths = get_data_paths('/home/thangdn3/Desktop/Expression-Net/DistillModel/'+train_label_path, train_path)

# no_test_img = len(image_paths_test)
no_train_img = len(image_paths)
train_steps=(no_train_img // BATCH_SIZE) + 1
# validation_steps=(no_test_img // BATCH_SIZE) + 1

opt = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
# opt = Adam(lr=lr)
model.compile(optimizer=opt, loss=losses)
model.fit_generator(
    generator.image_generator(image_paths,label_paths,preprocess_fn=preprocessing_function,target_size=TARGET_SIZE,batch_size = BATCH_SIZE),
    steps_per_epoch=train_steps,
    # steps_per_epoch=300,
    epochs=EPOCHS,
    initial_epoch=0,
    # validation_data=generator.image_generator(image_paths_test,label_paths_test,preprocess_fn=preprocessing_function,target_size=TARGET_SIZE, batch_size = BATCH_SIZE),
    # validation_steps=validation_steps,
    # validation_steps=300,
    callbacks=callbacks,
    use_multiprocessing=True)

# print('\nTesting ------------')
# # Evaluate the model with the metrics we defined earlier
# loss, accuracy = model.evaluate(x_test, y_test)

# print('test loss: ', loss)
# print('test accuracy: ', accuracy)

# Print the min max in fakequant
# for node in sess.graph.as_graph_def().node:
#     if 'weights_quant/AssignMaxLast' in node.name \
#         or 'weights_quant/AssignMinLast' in node.name:
#         tensor = sess.graph.get_tensor_by_name(node.name + ':0')
#         print('{} = {}'.format(node.name, sess.run(tensor)))

