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

import pickle
from tqdm import tqdm

import pose_utils as pu
import ST_model_nonTrainable_AlexNetOnFaces as Pose_model
sys.path.append('./kaffe')
sys.path.append('./ResNet')
from ThreeDMM_shape import ResNet_101 as resnet101_shape
from ThreeDMM_expr import ResNet_101 as resnet101_expr


_alexNetSize = 224 # for our model
# _alexNetSize = 227
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('image_size', _alexNetSize, 'Image side length.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
# tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size') #TODO test batch inference

inputlist = sys.argv[1] # You can try './input.csv' or input your own file


# Global parameters
_tmpdir = './tmp/'#save intermediate images needed to fed into ExpNet, ShapeNet, and PoseNet                                                                                                                                                       
print('> make dir')
if not os.path.exists( _tmpdir):
        os.makedirs( _tmpdir )
output_proc = 'output_preproc.csv' # save intermediate image list
factor = 0.25 # expand the given face bounding box to fit in the DCCNs

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
    side_length = max(w,h)
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
    crop_img = img_3[bbox_new[1][0]:bbox_new[3][0], bbox_new[0][0]:bbox_new[2][0], :]
    resized_crop_img = cv2.resize(crop_img, ( _alexNetSize, _alexNetSize ), interpolation = cv2.INTER_CUBIC)
    return resized_crop_img

import face_detection_opencv_dnn
def getFaceDetectionModel(DNN = "CAFFE"):    
    if DNN == "CAFFE":
        modelFile = "./res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "./deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "./opencv_face_detector_uint8.pb"
        configFile = "./opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    return net

def getFaceBBox(img, net, conf_threshold = 0.5):
    # rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # try:
    #     face_locations = face_recognition.face_locations(rgb_img)[0] # (top, right, bottom, left)
    # except Exception as e:
    #     print('Not detect bbox in this image')
    #     return None
    # y, x2, y2, x = face_locations    

    frameOpencvDnn = img.copy() #BGR frame
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)  #check original caffe model for these values

    net.setInput(blob)
    detections = net.forward()
    # bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x = int(detections[0, 0, i, 3] * frameWidth)
            y = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            # bboxes.append([x, y, x2, y2])
            return {'x': int(x), 'y': int(y), 'width': int(x2)-int(x), 'height': int(y2)-int(y)}
    print('Not detect bbox in this image')
    return None

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

def getFaceParams(img, face_detection_model):
    """
    input = image
    output = Shape + Expression + Pose + Texture info of the cropped face in the image
    """
    bbox_dict = getFaceBBox(img, face_detection_model)
    preprocessed_img = preProcessImage(img, bbox_dict)
                                                   
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
        
        # Fix the grey image                                                                                                                       
        if len(preprocessed_img.shape) < 3:
            image_r = np.reshape(preprocessed_img, (preprocessed_img.shape[0], preprocessed_img.shape[1], 1))
            image = np.append(image_r, image_r, axis=2)
            image = np.append(image, image_r, axis=2)
        else:
            image = preprocessed_img

        image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
        # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './merge_model')
        # use saved_model_cli tool on savedmodel to get input/output name
        # (Shape_Texture, Expr, Pose) = sess.run(['shapeCNN/shapeCNN_fc1/BiasAdd:0','exprCNN/exprCNN_fc1/BiasAdd:0','costs/add:0'], feed_dict={'Placeholder:0': image})
        (Shape_Texture, Expr, Pose) = sess.run([fc1ls, fc1le, pose_model.preds_unNormalized], feed_dict={x: image})

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
        # SEP,TEP = utils.projectBackBFM_withEP(model, Shape_Texture, Expr, Pose)
        # utils.write_ply_textureless(mesh_folder + '/TEST.ply', SEP, faces)
        # utils.write_ply(mesh_folder + '/TEST.ply', SEP, TEP, faces)
        return Shape_Texture, Expr, Pose, faces, model


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
                    # utils.write_ply_textureless(outFile + '_Shape_Expr_Pose.ply', SEP, faces)
                    utils.write_ply(outFile + '_Shape_Expr_Pose.ply', SEP, TEP, faces)

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

def prepare_distill_dataset(data_type = 'train', output_dir = './DistillModel/data/new_png'):
    tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch Size')
    list_path='./DistillModel/data/'+data_type+'_list.txt'
    list_name = os.path.basename(list_path)
    cur_dir = list_path[:-len(list_name)]
    output_img_dir = output_dir+'/preprocessed_img_'+data_type
    output_label_dir = output_dir+'/norm_label_'+data_type
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    text_file = open(list_path, "r")
    lines = text_file.read().split('\n')
    text_file.close()


    skip_list = []
    
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
    
    ###################
    # Face Pose-Net
    ###################
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

    ## Modifed Basel Face Model
    BFM_path = './Shape_Model/BaselFaceModel_mod.mat'
    model, faces = getBaselModel(BFM_path)

    face_detection_model = getFaceDetectionModel()
        
    ShapeScaler = StandardScaler()
    ExprScaler = StandardScaler()
    PoseScaler = StandardScaler()
    ps = []
    Shape_Texture = []
    Expr = []
    Pose = []
    config=tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
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
        print('============ Saving images files (and store labels before normalize)============')
        for i in tqdm(range(len(lines))):
            # paths=[]
            p = lines[i]
            if p == '': continue
            img_path = cur_dir+list_name[:-9]+'/'+p
            img = cv2.imread(img_path)
        
            bbox_dict = getFaceBBox(img, face_detection_model)
            if bbox_dict is None:
                skip_list.append(p)
                continue
            
            # paths.append(p)
            preprocessed_img = preProcessImage(img, bbox_dict) # still use bgr image as input for model
            # Fix the grey image                                                                                                                       
            if len(preprocessed_img.shape) < 3:
                image_r = np.reshape(preprocessed_img, (preprocessed_img.shape[0], preprocessed_img.shape[1], 1))
                image = np.append(image_r, image_r, axis=2)
                image = np.append(image, image_r, axis=2)
            else:
                image = preprocessed_img

            sub_dir = p.split('/')[0]
            base_name = os.path.basename(img_path)
            if not os.path.exists(output_img_dir+'/'+sub_dir):
                os.makedirs(output_img_dir+'/'+sub_dir)
            cv2.imwrite(output_img_dir+'/'+sub_dir+'/'+base_name[:-3]+'png',image,[cv2.IMWRITE_PNG_COMPRESSION, 0])

            image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
            # use saved_model_cli tool on savedmodel to get input/output name
            (cur_Shape_Texture, cur_Expr, cur_Pose) = sess.run([fc1ls, fc1le, pose_model.preds_unNormalized], feed_dict={x: image})


            ps += [p]
            Shape_Texture += [cur_Shape_Texture[0]]
            Expr += [cur_Expr[0]]
            Pose += [cur_Pose[0]]

        Shape_Texture = ShapeScaler.fit_transform(np.array(Shape_Texture))
        Expr = ExprScaler.fit_transform(np.array(Expr))
        Pose = PoseScaler.fit_transform(np.array(Pose))

        with open(output_label_dir+'/Shape_mean_'+data_type+'.bin', 'wb') as f:
            ShapeScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Shape_var_'+data_type+'.bin', 'wb') as f:
            ShapeScaler.var_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Expr_mean_'+data_type+'.bin', 'wb') as f:
            ExprScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Expr_var_'+data_type+'.bin', 'wb') as f:
            ExprScaler.var_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Pose_mean_'+data_type+'.bin', 'wb') as f:
            PoseScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Pose_var_'+data_type+'.bin', 'wb') as f:
            PoseScaler.var_.astype('>f8').tofile(f)
            
        print('============ Saving labels files ============')
        for i in tqdm(range(len(ps))):
            p = ps[i]
            sub_dir = p.split('/')[0]
            if not os.path.exists(output_label_dir+'/'+sub_dir):
                os.makedirs(output_label_dir+'/'+sub_dir)
            with open(output_label_dir+'/'+p[:-3] + 'dict', 'wb') as f:
                pickle.dump({'Shape_Texture':Shape_Texture[i],'Expr':Expr[i],'Pose':Pose[i]}, f)
    
    with open(cur_dir+'skip_list'+data_type+'.txt', 'w') as f:
        for item in skip_list:
            f.write("%s\n" % item)

# change train/test at three place
def prepare_batch_distill_dataset(data_type = 'train', output_dir = 'DistillModel/data/new', batch_size=128):
    list_path='DistillModel/data/'+data_type+'_list.txt'
    list_name = os.path.basename(list_path)
    cur_dir = list_path[:-len(list_name)]
    output_img_dir = output_dir+'/preprocessed_img_'+data_type
    output_label_dir = output_dir+'/norm_label_'+data_type
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    text_file = open(list_path, "r")
    lines = text_file.read().split('\n')
    text_file.close()


    skip_list = []
    init_op = tf.global_variables_initializer()
    ## Modifed Basel Face Model
    BFM_path = './Shape_Model/BaselFaceModel_mod.mat'
    model, faces = getBaselModel(BFM_path)

    face_detection_model = getFaceDetectionModel()

    tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Batch Size')
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
      
    # pose_model = getPoseNet(x)
    # x2, fc1ls = getShapeCNN(x)
    # fc1le = getExpressionCNN(x2)
                    
    # # Add ops to save and restore all the variables.                                                                                                                
    # init_op = tf.global_variables_initializer()
    # saver_pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Spatial_Transformer'))
    # saver_ini_shape_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shapeCNN'))
    # saver_ini_expr_net = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='exprCNN'))

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init_op)

        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './merge_model')
        # # Load face pose net model from Chang et al.'ICCVW17
        # load_path = "./fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt"
        # saver_pose.restore(sess, load_path)
        
        # # load 3dmm shape and texture model from Tran et al.' CVPR2017
        # load_path = "./Shape_Model/ini_ShapeTextureNet_model.ckpt"
        # saver_ini_shape_net.restore(sess, load_path)

        # # load our expression net model
        # load_path = "./Expression_Model/ini_exprNet_model.ckpt"
        # saver_ini_expr_net.restore(sess, load_path)
        
        ShapeScaler = StandardScaler()
        ExprScaler = StandardScaler()
        PoseScaler = StandardScaler()
        ps = []
        Shape_Texture = []
        Expr = []
        Pose = []
        print('============ Saving images files (and store labels before normalize)============')
        for i in tqdm(range(len(lines)//batch_size)):
            batch_images = None
            if i!= len(lines)//batch_size-1:
                size = batch_size
            else:
                size = len(lines)%batch_size

            paths=[]
            for j in range(size):
                p = lines[i*batch_size+j]
                if p == '': continue
                img_path = cur_dir+list_name[:-9]+'/'+p
                img = cv2.imread(img_path)
            


                bbox_dict = getFaceBBox(img, face_detection_model)
                if bbox_dict is None:
                    skip_list.append(p)
                    continue
                
                paths.append(p)
                preprocessed_img = preProcessImage(img, bbox_dict)
                # Fix the grey image                                                                                                                       
                if len(preprocessed_img.shape) < 3:
                    image_r = np.reshape(preprocessed_img, (preprocessed_img.shape[0], preprocessed_img.shape[1], 1))
                    image = np.append(image_r, image_r, axis=2)
                    image = np.append(image, image_r, axis=2)
                else:
                    image = preprocessed_img

                sub_dir = p.split('/')[0]
                base_name = os.path.basename(img_path)
                cv2.imwrite(output_img_dir+'/'+sub_dir+'/'+base_name[:-3]+'png',image,[cv2.IMWRITE_PNG_COMPRESSION, 0])

                image = np.reshape(image, [1, FLAGS.image_size, FLAGS.image_size, 3])
                if batch_images is None:
                    batch_images = image
                else:
                    batch_images = np.vstack((batch_images,image))
            
            filled_batch_images = np.zeros((batch_size, FLAGS.image_size, FLAGS.image_size, 3))
            filled_batch_images[:len(batch_images)]=batch_images
            # use saved_model_cli tool on savedmodel to get input/output name
            # (cur_Shape_Texture, cur_Expr, cur_Pose) = sess.run([fc1ls, fc1le, pose_model.preds_unNormalized], feed_dict={x: filled_batch_images})
            (cur_Shape_Texture, cur_Expr, cur_Pose) = sess.run(['shapeCNN/shapeCNN_fc1/BiasAdd:0','exprCNN/exprCNN_fc1/BiasAdd:0','costs/add:0'], feed_dict={'Placeholder:0': filled_batch_images})
        

            ps += [p]
            Shape_Texture += [cur_Shape_Texture]
            Expr += [cur_Expr]
            Pose += [cur_Pose]



        Shape_Texture = ShapeScaler.fit_transform(np.array(Shape_Texture))
        Expr = ExprScaler.fit_transform(np.array(Expr))
        Pose = PoseScaler.fit_transform(np.array(Pose))

        with open(output_label_dir+'/Shape_mean_'+data_type+'.bin', 'wb') as f:
            ShapeScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Shape_var_'+data_type+'.bin', 'wb') as f:
            ShapeScaler.var_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Expr_mean_'+data_type+'.bin', 'wb') as f:
            ExprScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Expr_var_'+data_type+'.bin', 'wb') as f:
            ExprScaler.var_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Pose_mean_'+data_type+'.bin', 'wb') as f:
            PoseScaler.mean_.astype('>f8').tofile(f)
        with open(output_label_dir+'/Pose_var_'+data_type+'.bin', 'wb') as f:
            PoseScaler.var_.astype('>f8').tofile(f)
            
        print('============ Saving labels files ============')
        for i in tqdm(range(len(ps))):
            p = ps[i]
            sub_dir = p.split('/')[0]
            if not os.path.exists(output_label_dir+'/'+sub_dir):
                os.makedirs(output_label_dir+'/'+sub_dir)
            with open(output_label_dir+'/'+p[:-3] + 'dict', 'wb') as f:
                pickle.dump({'Shape_Texture':Shape_Texture[i],'Expr':Expr[i],'Pose':Pose[i]}, f)
    
    with open(cur_dir+'skip_list'+data_type+'.txt', 'w') as f:
        for item in skip_list:
            f.write("%s\n" % item)

# TODO: put this inside 2 functions above
def prepare_preprocessed_dataset(list_path='DistillModel/data/train_list.txt', skip_list_path='DistillModel/data/skip_list.txt'):
    list_name = os.path.basename(list_path)
    cur_dir = list_path[:-len(list_name)]
    output_dir = cur_dir+'preprocessed_img_train'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    text_file = open(list_path, "r")
    full_list = text_file.read().split('\n')
    text_file.close()

    text_file = open(skip_list_path, "r")
    skip_list = text_file.read().split('\n')
    text_file.close()

    detected_list = list(set(full_list)^set(skip_list))

    skip_list2 = []
    for i in tqdm(range(len(detected_list))):
        p = detected_list[i]
        if p == '': continue
        img_path = cur_dir+list_name[:-9]+'/'+p
        img = cv2.imread(img_path)
    
        bbox_dict = getFaceBBox(img)
        if bbox_dict is None:
            skip_list2.append(p)
            continue
        preprocessed_img = preProcessImage(img, bbox_dict)

        sub_dir = p.split('/')[0]
        if not os.path.exists(output_dir+'/'+sub_dir):
            os.makedirs(output_dir+'/'+sub_dir)
        base_name = os.path.basename(img_path)
        cv2.imwrite(output_dir+'/'+sub_dir+'/'+base_name[:-3]+'png',preprocessed_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
    with open(cur_dir+'skip_list2.txt', 'w') as f:
        for item in skip_list2:
            f.write("%s\n" % item)
from sklearn.preprocessing import StandardScaler
def prepare_norm_labels(label_path, output_path, dataset = 'train'):
    def get_output(path):
        label_dict=None
        with open(path, 'rb') as dictionary_file:
            label_dict = pickle.load(dictionary_file)
        return label_dict['Shape_Texture'], label_dict['Expr'], label_dict['Pose']
    
    ShapeScaler = StandardScaler()
    ExprScaler = StandardScaler()
    PoseScaler = StandardScaler()

    label_paths = glob.glob(label_path+'/**/*.dict')
    Shape_Texture = []
    Expr = []
    Pose = []
    print('============ Loading files ============')
    for i in tqdm(range(len(label_paths))):
        p = label_paths[i]
    # for p in np.array(label_paths):
        (cur_Shape_Texture, cur_Expr, cur_Pose) = get_output(str(p))
        Shape_Texture += [cur_Shape_Texture]
        Expr += [cur_Expr]
        Pose += [cur_Pose]
    # Return a tuple of (input,output) to feed the network

    Shape_Texture = ShapeScaler.fit_transform(np.array(Shape_Texture))
    Expr = ExprScaler.fit_transform(np.array(Expr))
    Pose = PoseScaler.fit_transform(np.array(Pose))

    with open(output_path+'/Shape_mean_'+dataset+'.bin', 'wb') as f:
        ShapeScaler.mean_.astype('>f8').tofile(f)
    with open(output_path+'/Shape_var_'+dataset+'.bin', 'wb') as f:
        ShapeScaler.var_.astype('>f8').tofile(f)
    with open(output_path+'/Expr_mean_'+dataset+'.bin', 'wb') as f:
        ExprScaler.mean_.astype('>f8').tofile(f)
    with open(output_path+'/Expr_var_'+dataset+'.bin', 'wb') as f:
        ExprScaler.var_.astype('>f8').tofile(f)
    with open(output_path+'/Pose_mean_'+dataset+'.bin', 'wb') as f:
        PoseScaler.mean_.astype('>f8').tofile(f)
    with open(output_path+'/Pose_var_'+dataset+'.bin', 'wb') as f:
        PoseScaler.var_.astype('>f8').tofile(f)
        
    print('============ Saving files ============')
    for i in tqdm(range(len(label_paths))):
        p = label_paths[i]
        if p == '': continue

        sub_dir = p[-20:-13]
        if not os.path.exists(output_path+'/'+sub_dir):
            os.makedirs(output_path+'/'+sub_dir)
        with open(output_path+'/'+p[-20:], 'wb') as f:
            pickle.dump({'Shape_Texture':Shape_Texture[i],'Expr':Expr[i],'Pose':Pose[i]}, f)

import copy
def main(_):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    
    # print dev
    # with tf.device(dev):
    #     extract_PSE_feats()

    # img = cv2.imread('./images/Disgust_71_1.jpg')
    # img = cv2.imread('./images/Happy_183_1.jpg')
    # face_detection_model = getFaceDetectionModel()
    # with tf.device(dev):
    #     Shape_Texture, Expr, Pose, faces, model = getFaceParams(img, face_detection_model)
    #     SEP,TEP = utils.projectBackBFM_withEP(model, Shape_Texture, Expr, Pose)
        # output_ply = getPlyFile(SEP, TEP, faces)
        # print(output_ply)

    with tf.device(dev):
        # saveMergeModel()
        # prepare_distill_dataset(data_type='test')
        prepare_distill_dataset(data_type='test')
        # prepare_distill_dataset()

    # prepare_preprocessed_dataset()
    # with tf.device(dev):
    #     Shape_Texture, Expr, Pose, faces, model = getFaceParams(img)
    #     base_S = utils.get_base_S(model)
    #     dtype='>f8'
    #     utils.array_to_binary(base_S, './base_S.bin',dtype=dtype)
    #     utils.array_to_binary(model.expEV, './model_expEV.bin',dtype=dtype)
    #     utils.array_to_binary(model.expPC, './model_expPC.bin',dtype=dtype)
    #     utils.array_to_binary(model.shapeEV, './model_shapeEV.bin',dtype=dtype)
    #     utils.array_to_binary(model.shapePC, './model_shapePC.bin',dtype=dtype)

    #     name = "Disgust"
    #     utils.array_to_binary(Pose, './'+name+'_Pose.bin',dtype=dtype)
    #     utils.array_to_binary(Expr, './'+name+'_Expr.bin',dtype=dtype)
    #     utils.array_to_binary(Shape_Texture, './'+name+'_Shape_Texture.bin',dtype=dtype)

    #     with open('./base_S.bin', 'rb') as f:
    #         base_S_load = np.fromfile(f, dtype=dtype)
    #     with open('./model_expEV.bin', 'rb') as f:
    #         model_expEV = np.fromfile(f, dtype=dtype)
    #     with open('./model_expPC.bin', 'rb') as f:
    #         model_expPC = np.fromfile(f, dtype=dtype)
    #         model_expPC = np.reshape(model_expPC,(-1,29))
    #     with open('./model_shapeEV.bin', 'rb') as f:
    #         model_shapeEV = np.fromfile(f, dtype=dtype)
    #     with open('./model_shapePC.bin', 'rb') as f:
    #         model_shapePC = np.fromfile(f, dtype=dtype)
    #         model_shapePC = np.reshape(model_shapePC,(-1,99))

    #     with open('./'+name+'_Pose.bin', 'rb') as f:
    #         Pose_load = np.fromfile(f, dtype=dtype)
    #     with open('./'+name+'_Expr.bin', 'rb') as f:
    #         Expr_load = np.fromfile(f, dtype=dtype)
    #     with open('./'+name+'_Shape_Texture.bin', 'rb') as f:
    #         Shape_Texture_load = np.fromfile(f, dtype=dtype)

    #     SEP,TEP,d0 = utils.projectBackBFM_withEP(copy.deepcopy(model), copy.deepcopy(Shape_Texture), copy.deepcopy(Expr), copy.deepcopy(Pose))
    #     SEP_calc_by_update,d1 = utils.update_pose_expr(model_expEV, model_expPC, model_shapeEV, model_shapePC, base_S_load, Expr_load, Pose_load, Shape_Texture_load)
        
    #     #TODO save and load make array different from original.
    #     # print(np.array_equal(d0['S_RT'], d1['S_RT']))
    #     # print('====================')
    #     # print(np.array_equal(d0['E'], d1['E']))
 
    #     utils.write_ply(mesh_folder + '/Happy_without_update.ply', SEP, TEP, faces)
    #     utils.write_ply(mesh_folder + '/Happy_with_update.ply', SEP_calc_by_update, TEP, faces)
    #     pass

if __name__ == '__main__':
    tf.app.run()
