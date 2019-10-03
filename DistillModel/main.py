import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import RMSprop
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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

BATCH_SIZE = 16
TARGET_SIZE=(224, 224)
EPOCHS = 100
NO_GPU = 2

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

opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss=losses, metrics=['mse'])
filepath="./models/MobileNetV2-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks = [checkpointer, tensorboard_callback]
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

train_path = 'data/test'
test_path = 'data/test'
train_label_path = 'data/label_test'
test_label_path = 'data/label_test'

no_test_img = len(list(paths.list_images(test_path)))
no_train_img = len(list(paths.list_images(train_path)))
train_steps=(no_train_img // BATCH_SIZE) + 1
validation_steps=(no_test_img // BATCH_SIZE) + 1

skip_list = []
image_paths = []
label_paths = []
with open('./data/skip_list_test.txt') as fp:
    skip_list = fp.readlines()
    skip_list = [l.rstrip() for l in skip_list]

with open('./data/test_list.txt') as fp:
    lines = fp.readlines()
    lines = [l.rstrip() for l in lines]

    skipped_line = list(set(lines) ^ set(skip_list))
    for p in skipped_line:
        image_paths.append(train_path + '/' + p)
        label_paths.append(train_label_path + '/' + p[:-3]+'dict')

model.fit_generator(
    generator.image_generator(image_paths,label_paths,preprocess_fn=preprocessing_function,target_size=TARGET_SIZE,batch_size = BATCH_SIZE),
    steps_per_epoch=train_steps,
    epochs=EPOCHS,
    initial_epoch=0,
    # validation_data=generator.image_generator(image_paths,label_paths,preprocess_fn=preprocessing_function,target_size=TARGET_SIZE, batch_size = BATCH_SIZE),
    # validation_steps=validation_steps,
    callbacks=callbacks)

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

