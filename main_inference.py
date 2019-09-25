import sys
import numpy as np
import tensorflow as tf
import cv2
import scipy.io as sio
import os
import os.path
import glob
import time
import scipy
import scipy.io as sio
import utils
import myparse
import csv


import pose_utils as pu
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model
sys.path.append('./kaffe')
sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', 227, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')

inputlist = sys.argv[1] # You can try './input.csv' or input your own file


# Global parameters
_tmpdir = './tmp/'#save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet                                                                                                                                                       
print('> make dir')
if not os.path.exists( _tmpdir):
        os.makedirs( _tmpdir )
output_proc = 'output_preproc.csv' # save intermediate image list
factor = 0.25 # expand the given face bounding box to fit in the DCCNs
_alexNetSize = 227

mesh_folder = './output_ply' # The location where .ply files are saved
if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)


# Get training image/labels mean/std for pose CNN
file = np.load("./fpn_new_model/perturb_Oxford_train_imgs_mean.npz")
train_mean_vec = file["train_mean_vec"] # [0,1]
del file
file = np.load("./fpn_new_model/perturb_Oxford_train_labels_mean_std.npz")
mean_labels = file["mean_labels"]
std_labels = file["std_labels"]
del file

# Get training image mean for Shape CNN
mean_image_shape = np.load('./Shape_Model/3DMM_shape_mean.npy') # 3 x 224 x 224 
mean_image_shape = np.transpose(mean_image_shape, [1,2,0]) # 224 x 224 x 3, [0,255]


# Get training image mean for Expression CNN
mean_image_exp = np.load('./Expression_Model/3DMM_expr_mean.npy') # 3 x 224 x 224
mean_image_exp = np.transpose(mean_image_exp, [1,2,0]) # 224 x 224 x 3, [0,255]

def preProcessImage(im, bbox_dict): #cv2 image (bgr)
    sys.stdout.flush()
    lt_x = bbox_dict['x']
    lt_y = bbox_dict['y']
    rb_x = lt_x + bbox_dict['width']
    rb_y = lt_y + bbox_dict['height']
    w = bbox_dict['width']
    h = bbox_dict['height']
    center = ( (lt_x+rb_x)/2, (lt_y+rb_y)/2 )
    side_length = max(w,h);
    bbox = np.zeros( (4,1), dtype=np.float32 )
    bbox[0] = center[0] - side_length/2
    bbox[1] = center[1] - side_length/2
    bbox[2] = center[0] + side_length/2
    bbox[3] = center[1] + side_length/2
    #%% Get the expanded square bbox
    bbox_red = pu.increaseBbox(bbox, factor)
    img_3, bbox_new = pu.image_bbox_processing_v2(im, bbox_red)
    #%% Crop and resized
    bbox_new =  np.ceil( bbox_new )
    side_length = max( bbox_new[2] - bbox_new[0], bbox_new[3] - bbox_new[1] )
    bbox_new[2:4] = bbox_new[0:2] + side_length
    #crop_img = img(bbox_red(2):bbox_red(4), bbox_red(1):bbox_red(3), :);
    #resized_crop_img = imresize(crop_img, [227, 227]);# % re-scaling to 227 x 227
    bbox_new = bbox_new.astype(int)
    crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :];
    resized_crop_img = cv2.resize(crop_img, ( _alexNetSize, _alexNetSize ), interpolation = cv2.INTER_CUBIC)
    return resized_crop_img

# TODO implement MTCNN later
import face_recognition
def getFaceBBox(img):
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)[0] # (top, right, bottom, left)
    y, x2, y2, x = face_locations
    return {'x': int(x), 'y': int(y), 'width': int(x2)-int(x), 'height': int(y2)-int(y)}

def getPlyFile(S, T, faces):
	nV = S.shape[0]
	nF = faces.shape[0]
	output = ''
	output += 'ply\n'
	output += 'format ascii 1.0\n'
	output += 'element vertex ' + str(nV) + '\n'
	output += 'property float x\n'
	output += 'property float y\n'
	output += 'property float z\n'
	output += 'property uchar red\n'
	output += 'property uchar green\n'
	output += 'property uchar blue\n'
	output += 'element face ' + str(nF) + '\n'
	output += 'property list uchar int vertex_indices\n'
	output += 'end_header\n'

	for i in range(0,nV):
		output += '%0.4f %0.4f %0.4f %d %d %d\n' % (S[i,0],S[i,1],S[i,2], int(T[i,0]), int(T[i, 1]), int(T[i, 2]))

	for i in range(0,nF):
		output += '3 %d %d %d\n' % (faces[i,0],faces[i,1],faces[i,2])
    
	return output

def getFaceParams(img):
    """
    input = image
    output = Shape + Expression + Pose + Texture info of the cropped face in the image
    """
    bbox_dict = getFaceBBox(img)
    preprocessed_img = preProcessImage(img, bbox_dict)
                                                                                                                         
    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)
        
        ## Modifed Basel Face Model
        BFM_path = './Shape_Model/BaselFaceModel_mod.mat'
        model, faces = getBaselModel(BFM_path)
        
        # Fix the grey image                                                                                                                       
        if len(preprocessed_img.shape) < 3:
            image_r = np.reshape(preprocessed_img, (preprocessed_img.shape[0], preprocessed_img.shape[1], 1))
            image = np.append(image_r, image_r, axis=2)
            image = np.append(image, image_r, axis=2)
        else:
            image = preprocessed_img

        image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './merge_model')
        # use saved_model_cli tool on savedmodel to get input/output name
        (Shape_Texture, Expr, Pose) = sess.run(['shapeCNN/shapeCNN_fc1/BiasAdd:0','exprCNN/exprCNN_fc1/BiasAdd:0','costs/add:0'], feed_dict={'Placeholder:0': image})
        
        Pose = np.reshape(Pose, [-1])
        Shape_Texture = np.reshape(Shape_Texture, [-1])
        Shape = Shape_Texture
    #     Shape = Shape_Texture[0:99]
        Shape = np.reshape(Shape, [-1])
        Expr = np.reshape(Expr, [-1])

        #########################################
        ### Save 3D shape information (.ply file)
        #########################################
        # Shape Only
        #S,T = utils.projectBackBFM(model,Shape_Texture)
        #utils.write_ply_textureless(outFile + '_ShapeOnly.ply', S, faces)

        
        # Shape + Expression
        #SE,TE = utils.projectBackBFM_withExpr(model, Shape_Texture, Expr)
        #utils.write_ply_textureless(outFile + '_Shape_and_Expr.ply', SE, faces)
        
        # Shape + Expression + Pose
        SEP,TEP = utils.projectBackBFM_withEP(model, Shape_Texture, Expr, Pose)
        # utils.write_ply_textureless(mesh_folder + '/TEST.ply', SEP, TEP, faces)
        output_ply = getPlyFile(SEP, TEP, faces)
        print(output_ply)


def getPoseNet(input_placeholder):
    try:
        net_data = np.load("./fpn_new_model/PAM_frontal_ALexNet.npy").item()
    except Exception as e:
        net_data = np.load("./fpn_new_model/PAM_frontal_ALexNet.npy", allow_pickle=True).item()

    pose_labels = np.zeros([FLAGS.batch_size,6])
    x1 = tf.image.resize_bilinear(input_placeholder, tf.constant([227,227], dtype=tf.int32))
    
    # Image normalization
    x1 = x1 / 255. # from [0,255] to [0,1]
    # subtract training mean
    mean = tf.reshape(train_mean_vec, [1, 1, 1, 3])
    mean = tf.cast(mean, 'float32')
    x1 = x1 - mean

    pose_model = Pose_model.Pose_Estimation(x1, pose_labels, 'valid', 0, 1, 1, 0.01, net_data, FLAGS.batch_size, mean_labels, std_labels)
    pose_model._build_graph()
    del net_data
    return pose_model

def getShapeCNN(input_placeholder):
    """
    output: norm input and Shape vector
    """
    x2 = tf.image.resize_bilinear(input_placeholder, tf.constant([224,224], dtype=tf.int32))
    x2 = tf.cast(x2, 'float32')
    x2 = tf.reshape(x2, [FLAGS.batch_size, 224, 224, 3])
    
    # Image normalization
    mean = tf.reshape(mean_image_shape, [1, 224, 224, 3])
    mean = tf.cast(mean, 'float32')
    x2 = x2 - mean

    with tf.variable_scope('shapeCNN'):
        net_shape = resnet101_shape({'input': x2}, trainable=True)
        pool5 = net_shape.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size,-1])

        
        npzfile = np.load('./ResNet/ShapeNet_fc_weights.npz')
        ini_weights_shape = npzfile['ini_weights_shape']
        ini_biases_shape = npzfile['ini_biases_shape']
        with tf.variable_scope('shapeCNN_fc1'):
            fc1ws = tf.Variable(tf.reshape(ini_weights_shape, [2048,-1]), trainable=True, name='weights')
            fc1bs = tf.Variable(tf.reshape(ini_biases_shape, [-1]), trainable=True, name='biases')
            fc1ls = tf.nn.bias_add(tf.matmul(pool5, fc1ws), fc1bs)
    return x2, fc1ls

def getExpressionCNN(norm_input):
    with tf.variable_scope('exprCNN'):
        net_expr = resnet101_expr({'input': norm_input}, trainable=True)
        pool5 = net_expr.layers['pool5']
        pool5 = tf.squeeze(pool5)
        pool5 = tf.reshape(pool5, [FLAGS.batch_size,-1])

        
        npzfile = np.load('./ResNet/ExpNet_fc_weights.npz')
        ini_weights_expr = npzfile['ini_weights_expr']
        ini_biases_expr = npzfile['ini_biases_expr']
        with tf.variable_scope('exprCNN_fc1'):
            fc1we = tf.Variable(tf.reshape(ini_weights_expr, [2048,29]), trainable=True, name='weights')
            fc1be = tf.Variable(tf.reshape(ini_biases_expr, [29]), trainable=True, name='biases')
            fc1le = tf.nn.bias_add(tf.matmul(pool5, fc1we), fc1be)
    return fc1le

def getBaselModel(BFM_path):
    model = scipy.io.loadmat(BFM_path,squeeze_me=True,struct_as_record=False)
    model = model["BFM"]
    faces = model.faces-1
    return model, faces

def extract_PSE_feats():
	# Prepare data
        data_dict = myparse.parse_input(inputlist) # please see input.csv for the input format
        print(len(data_dict))
        ## Pre-processing the images                                                                                                                                                                              
        print('> preproc')
        pu.preProcessImage(_tmpdir, data_dict, './', factor, _alexNetSize, output_proc)


	# placeholders for the batches                                                                                                                            
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
      
        # ###################
        # # Face Pose-Net
        # ###################
        pose_model = getPoseNet(x)

        ###################
        # Shape CNN
        ###################
        x2, fc1ls = getShapeCNN(x)                       

        ###################
        # Expression CNN
        ###################
        fc1le = getExpressionCNN(x2)
                       
        # Add ops to save and restore all the variables.                                                                                                                
        init_op = tf.global_variables_initializer()
        saver_pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Spatial_Transformer'))
        saver_ini_shape_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
        saver_ini_expr_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(init_op)

            # Load face pose net model from Chang et al.'ICCVW17
            load_path = "./fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt"
            saver_pose.restore(sess, load_path)

            
            # load 3dmm shape and texture model from Tran et al.' CVPR2017
            load_path = "./Shape_Model/ini_ShapeTextureNet_model.ckpt"
            saver_ini_shape_net.restore(sess, load_path)

            # load our expression net model
            load_path = "./Expression_Model/ini_exprNet_model.ckpt"
            saver_ini_expr_net.restore(sess, load_path)
            
            ## Modifed Basel Face Model
            BFM_path = './Shape_Model/BaselFaceModel_mod.mat'
            model, faces = getBaselModel(BFM_path)
            print('> Loaded the Basel Face Model to write the 3D output!')

            print('> Start to estimate Expression, Shape, and Pose!')
            with open(output_proc, 'rb') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                for row in csvreader:

                    image_key = row[0]
                    image_file_path = row[1]

                    print('> Process ' + image_file_path)
                    
                    image = cv2.imread(image_file_path,1) # BGR                                                                                  
                    image = np.asarray(image)
                    # Fix the grey image                                                                                                                       
                    if len(image.shape) < 3:
                        image_r = np.reshape(image, (image.shape[0], image.shape[1], 1))
                        image = np.append(image_r, image_r, axis=2)
                        image = np.append(image, image_r, axis=2)

                    image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
                    (Shape_Texture, Expr, Pose) = sess.run([fc1ls, fc1le, pose_model.preds_unNormalized], feed_dict={x: image})

                    outFile = mesh_folder + '/' + image_key
                    
                    Pose = np.reshape(Pose, [-1])
                    Shape_Texture = np.reshape(Shape_Texture, [-1])
                    Shape = Shape_Texture
                #     Shape = Shape_Texture[0:99]
                    Shape = np.reshape(Shape, [-1])
                    Expr = np.reshape(Expr, [-1])

                    #########################################
                    ### Save 3D shape information (.ply file)
                    #########################################
                    # Shape Only
                    #S,T = utils.projectBackBFM(model,Shape_Texture)
                    #utils.write_ply_textureless(outFile + '_ShapeOnly.ply', S, faces)

                    
                    # Shape + Expression
                    #SE,TE = utils.projectBackBFM_withExpr(model, Shape_Texture, Expr)
                    #utils.write_ply_textureless(outFile + '_Shape_and_Expr.ply', SE, faces)
                    
                    # Shape + Expression + Pose
                    SEP,TEP = utils.projectBackBFM_withEP(model, Shape_Texture, Expr, Pose)
                    utils.write_ply_textureless(outFile + '_Shape_Expr_Pose.ply', SEP, TEP, faces)

def saveMergeModel():                                                                    
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    
    # ###################
    # # Face Pose-Net
    # ###################
    pose_model = getPoseNet(x)

    ###################
    # Shape CNN
    ###################
    x2, fc1ls = getShapeCNN(x)                       

    ###################
    # Expression CNN
    ###################
    fc1le = getExpressionCNN(x2)
                    
    # Add ops to save and restore all the variables.                                                                                                                
    init_op = tf.global_variables_initializer()
    saver_pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Spatial_Transformer'))
    saver_ini_shape_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
    saver_ini_expr_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        # Load face pose net model from Chang et al.'ICCVW17
        load_path = "./fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt"
        saver_pose.restore(sess, load_path)
        
        # load 3dmm shape and texture model from Tran et al.' CVPR2017
        load_path = "./Shape_Model/ini_ShapeTextureNet_model.ckpt"
        saver_ini_shape_net.restore(sess, load_path)

        # load our expression net model
        load_path = "./Expression_Model/ini_exprNet_model.ckpt"
        saver_ini_expr_net.restore(sess, load_path)

        tf.saved_model.simple_save(sess,
                                   "./merge_model",
                                   inputs={"x": x},
                                   outputs={"Shape_Texture"    :fc1ls,
                                            "Expr"             :fc1le,
                                            "Pose"             :pose_model.preds_unNormalized})

def main(_):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    
    # print dev
    # with tf.device(dev):
    #     extract_PSE_feats()

    img = cv2.imread('./images/Happy_183_1.jpg')
    img = np.asarray(img)
    with tf.device(dev):
        getFaceParams(img)
        # saveMergeModel()


if __name__ == '__main__':
    tf.app.run()
