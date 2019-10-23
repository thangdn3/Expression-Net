import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.platform import gfile
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocessing_function
from imutils import paths
import numpy as np
import glob
from main import get_pretrained_model, load_pb_model, pb_model_predict
from main import TARGET_SIZE
import generator

model = get_pretrained_model()
model.load_weights('./DistillModel/models/Pose_MobileNetV2-97-0.01.hdf5')
model.summary()

def get_one_batch(BATCH_SIZE):
    batch_x = []
    batch_y = []
    image_paths = [
        '/home/duongnhatthang/Desktop/Projects/Expression-Net/DistillModel/data/preprocessed/n000001/0001_01.jpg',
        '/home/duongnhatthang/Desktop/Projects/Expression-Net/DistillModel/data/preprocessed/n000001/0002_01.jpg',
        '/home/duongnhatthang/Desktop/Projects/Expression-Net/DistillModel/data/preprocessed/n000001/0003_01.jpg',
        '/home/duongnhatthang/Desktop/Projects/Expression-Net/DistillModel/data/preprocessed/n000001/0004_01.jpg']

    for input_path in np.array(image_paths):
        input_img = generator.get_input(input_path)        
        input_img = generator.resize(input_img,TARGET_SIZE)
        input_img = preprocessing_function(input_img)
        batch_x += [ input_img ]
    batch_x = np.array(batch_x)

    # train_path = 'data/test'
    # train_label_path = 'data/label_test'

    # skip_list = []
    # image_paths = []
    # label_paths = []
    # with open('./data/skip_list_test.txt') as fp:
    #     skip_list = fp.readlines()
    #     skip_list = [l.rstrip() for l in skip_list]

    # with open('./data/test_list.txt') as fp:
    #     lines = fp.readlines()
    #     lines = [l.rstrip() for l in lines]

    #     skipped_line = list(set(lines) ^ set(skip_list))
    #     for p in skipped_line:
    #         image_paths.append(train_path + '/' + p)
    #         label_paths.append(train_label_path + '/' + p[:-3]+'dict')

    # BATCH_SIZE = 4
    # image_paths,label_paths
    # batch_x, batch_y = generator.get_one_batch(image_paths,label_paths,preprocess_fn=preprocessing_function,target_size=TARGET_SIZE,batch_size = BATCH_SIZE),
    return batch_x, batch_y

batch_x, batch_y = get_one_batch(BATCH_SIZE=4)
keras_predictions = model.predict(batch_x)

sess = load_pb_model("./DistillModel/models/Pose_MobileNetV2-97-0.01.pb")
tensorflow_predictions = pb_model_predict(sess, batch_x, input_tensor_name='import/input_1:0',
    output_tensor_names='import/Pose/BiasAdd:0')
# tensorflow_predictions = pb_model_predict(sess, batch_x, input_tensor_name='import/input_1:0', 
#     output_tensor_names=['import/Shape_Texture/BiasAdd:0','import/Expr/BiasAdd:0','import/Pose/BiasAdd:0'])

if (tensorflow_predictions == keras_predictions).all():
    print("keras_predictions is the same as tensorflow_predictions")
else:
    print("keras_predictions is NOT the same as tensorflow_predictions")
pass


# Convert to tflite

# tflite_convert \
#   --output_file=./DistillModel/random_model.tflite \
#   --graph_def_file=./model/tf_model.pb \
#   --input_shapes=1,224,224,3 \
#   --input_arrays=input_1 \
#   --output_arrays=Shape_Texture/BiasAdd,Expr/BiasAdd,Pose/BiasAdd



# Convert to tflite with code (untested)

# import tensorflow as tf
# from main import TARGET_SIZE

# graph_def_file = "./model/tf_model.pb"
# input_arrays = ["input_1"]
# output_arrays = ["Shape_Texture/BiasAdd","Expr/BiasAdd","Pose/BiasAdd"]

# converter = tf.lite.TFLiteConverter.from_frozen_graph(
#         graph_def_file, input_arrays, output_arrays,input_shapes={"image_tensor":[1,TARGET_SIZE[0],TARGET_SIZE[1],3]})
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)