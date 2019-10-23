# from skimage.io import imread
import numpy as np
import pickle
import cv2
# from skimage.transform import resize

def get_input(path):    
    # img = imread(path)    
    img = cv2.imread(path)
    rgb_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    return rgb_img

def get_output(path):
    label_dict=None
    with open(path, 'rb') as dictionary_file:
        label_dict = pickle.load(dictionary_file)
    return label_dict['Shape_Texture'], label_dict['Expr'], label_dict['Pose']

def get_one_batch(image_paths,label_paths,preprocess_fn, target_size, batch_size = 64):
    # Select files (paths/indices) for the batch
    indexes = np.random.choice(a = len(image_paths), size = batch_size)
    batch_input = []
    batch_output = [] 
    
    # Read in each input, perform preprocessing and get labels
    for input_path in np.array(image_paths)[indexes]:
        input_img = get_input(input_path)        
        # input_img = resize(input_img,target_size)
        input_img = preprocess_fn(input_img)
        batch_input += [ input_img ]

    Shape_Texture = []
    Expr = []
    Pose = []
    for output_path in np.array(label_paths)[indexes]:
        (cur_Shape_Texture, cur_Expr, cur_Pose) = get_output(str(output_path))
        Shape_Texture += [cur_Shape_Texture]
        Expr += [cur_Expr]
        Pose += [cur_Pose]
    # Return a tuple of (input,output) to feed the network
    batch_x = np.array( batch_input )
    batch_y = {'Shape_Texture':np.array(Shape_Texture),'Expr':np.array(Expr),'Pose':np.array(Pose)}
    return batch_x, batch_y

def image_generator(image_paths,label_paths,preprocess_fn, target_size, batch_size = 64):
    while True:
        batch_x, batch_y = get_one_batch(image_paths,label_paths,preprocess_fn, target_size, batch_size)
    
        yield( batch_x, batch_y )