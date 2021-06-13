import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import random
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO VIDEO DIRECTORY
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/YawDD/YawDD dataset/Mirror/Female/18-FemaleNoGlasses-Yawning.avi'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/YawDD/YawDD dataset/Mirror/Female/31-FemaleGlasses-Normal.avi'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/YawDD/YawDD dataset/Mirror/Female/16-FemaleGlasses-Yawning.avi'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/YawDD/YawDD dataset/Dash/Female/11-FemaleGlasses.avi.avi'
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import pathlib
import numpy as np
from PIL import Image
import warnings



# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/mydataset/sukran_sunglasses.mp4'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'exported-models_bitirme_yeni/my_mobilenet_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'exported-models_bitirme_yeni/my_mobilenet_model/saved_model/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.41

# Load the model
# ~~~~~~~~~~~~~~


PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
        
def zerolist(n):
    listofzeros = [0] * n
    return listofzeros

for sira in range(1,384):
        
    VIDEO_PATHS = 'C:/Users/kumralf/Desktop/bitirme/PROJE_DOKUMANLAR/videos_tek/'+str(sira)+'.avi'

    pathson = pathlib.PurePath(VIDEO_PATHS)
    frame_no=0
    matrix = [[0] * 50 for i in range(10)]
    karar= zerolist(50)
    f=0
    it=0
    print('Running inference for {}... '.format(VIDEO_PATHS), end='')

    video = cv2.VideoCapture(VIDEO_PATHS)
    while(video.isOpened()):
    
    
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)


    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        scores = detections['detection_scores']
        names = detections['detection_classes']
        detection_names = names[scores > MIN_CONF_THRESH]
    # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        frame_with_detections = frame.copy()
    
        frame_no+=1 
        for r in range(0,8):
            if r in detection_names:
                matrix[frame_no][(r*6):(r*6+3)] = [1,1,1]
        matrix.append(zerolist(50))


        tekrar = 50    # x framede bir kontrol
        threshold = tekrar*0.7

        matris=np.copy(matrix)
        matris=np.delete(matris,np.s_[len(matris)-10:len(matris)],axis=0)
    
        if frame_no > 49:
            matris = matris[(f):len(matris)][:]
            f+=1
        
            sonuc = matris.sum(axis=0)
        # sonuc = np.asmatrix(sonuc)
            karar = [1 if sonuc > round(threshold) else 0 for sonuc in sonuc]
            if (sonuc[4*6]>tekrar*0.25):
                karar[4*6]=1
                karar[4*6+1]=1
                karar[4*6+2]=1
                # karar[4*6+3]=1
            elif (sonuc[5*6]>tekrar*0.2):
                karar[5]=1
                karar[5*6+1]=1
                karar[5*6+2]=1
                # karar[5*6+3]=1
            elif (sonuc[6*6]>tekrar*0.2):
                karar[6]=1
                karar[6*6+1]=1
                karar[6*6+2]=1
                # karar[6*6+3]=1

        
            # if (karar==[1,0,0,1,0,0,0]):
            if (karar[1*6]==1 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==1 and karar[5*6]==0 and karar[6*6]==0 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/normal/normal_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[1,0,1,0,0,0,0]):
            elif (karar[1*6]==1 and karar[2*6]==0 and karar[3*6]==1 and karar[4*6]==0 and karar[5*6]==0 and karar[6*6]==0 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/normal/normal_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[1,0,0,0,0,0,0]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/normal/normal_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,0,1,0,0,1,0]):
            elif (karar[1*6]==0 and karar[2*6]==0 and karar[3*6]==1 and karar[4*6]==0 and karar[5*6]==0 and karar[6*6]==1 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/normal/normal_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0 
            # elif (karar==[0,0,0,1,0,1,0]):
            elif (karar[1*6]==0 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==1 and karar[5*6]==0 and karar[6*6]==1 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/normal/normal_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[1,0,0,0,1,0,0]):
            elif (karar[1*6]==1 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==0 and karar[5*6]==1 and karar[6*6]==0 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/drowsy/drowsy_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[0,0,0,0,1,1,0]):
            elif (karar[1*6]==0 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==0 and karar[5*6]==1 and karar[6*6]==1 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/drowsy/drowsy_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[0,0,0,0,1,0,0]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/drowsy/drowsy_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,1,0,1,0,0,0]):
            elif (karar[1*6]==0 and karar[2*6]==1 and karar[3*6]==0 and karar[4*6]==1 and karar[5*6]==0 and karar[6*6]==0 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[0,1,0,0,0,0,0]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,1,0,0,1,0,0]):
            elif (karar[1*6]==0 and karar[2*6]==1 and karar[3*6]==0 and karar[4*6]==0 and karar[5*6]==1 and karar[6*6]==0 and karar[7*6]==0):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[0,1,1,0,0,0,0]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,0,0,0,0,1,1]):
            elif (karar[1*6]==0 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==0 and karar[5*6]==0 and karar[6*6]==1 and karar[7*6]==1):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
            # elif (karar==[0,0,0,1,0,1,1]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,1,0,1,0,0,1]):
            #     matris = (matris * 255).astype(np.uint8)
            #     Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
            #     matris=0
            # elif (karar==[0,0,0,0,0,0,1]):
            elif (karar[1*6]==0 and karar[2*6]==0 and karar[3*6]==0 and karar[4*6]==0 and karar[5*6]==0 and karar[6*6]==0 and karar[7*6]==1):
                matris = (matris * 255).astype(np.uint8)
                Image.fromarray(matris, mode='L').save('C:/Users/kumralf/Desktop/makale/features50/danger/danger_' + str(pathson.name)+'_'+str(frame_no) + '.jpg')
                matris=0
                
                
                
            # matris = (matris * 255).astype(np.uint8)
            # Image.fromarray(matris, mode='L').save('pic' + str(frame_no) + '.jpg')
            # matris=0
            
            # print("\n___%d___" %(it))
            # print("sonuc: ",sonuc)
            # print("karar: ",karar)
        

    
    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
        viz_utils.visualize_boxes_and_labels_on_image_array(
              frame_with_detections,
              detections['detection_boxes'],
              detections['detection_classes'],
              detections['detection_scores'],
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              line_thickness=1,
              min_score_thresh=MIN_CONF_THRESH,
              agnostic_mode=False)
        
        cv2.imshow('Object Detector', frame_with_detections)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Done")
    print("frame_no: ", frame_no)



