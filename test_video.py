import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import pathlib
from pygame import mixer
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from some_functions import kararver
from some_functions import zerolist
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO VIDEO DIRECTORY
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/YawDD/YawDD dataset/Mirror/Female/1-FemaleNoGlasses-Yawning.avi'
VIDEO_PATHS = 'C:/Users/kumralf/Desktop/bitirme/PROJE_DOKUMANLAR/YawDD/YawDD dataset/Mirror/Female/27-FemaleSunGlasses-Yawning.avi'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/yawdd_male.mp4'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/merge_videos/35.mp4'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/mydataset/beyza_sunglasses.mp4'
# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/mydataset/oksan_sunglasses.mp4'

frame_size =(640,480)

# out = cv2.VideoWriter('C:/Users/kumralf/Desktop/mydataset/bitirme_output/yawdd_female.mp4', -1, 20.0, frame_size)

pathson = pathlib.PurePath(VIDEO_PATHS)

# VIDEO_PATHS = 'C:/Users/kumralf/Desktop/mydataset/sukran_sunglasses.mp4'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'exported-models_bitirme_yeni/my_mobilenet_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'exported-models_bitirme_yeni/my_mobilenet_model/saved_model/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = 0.5

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
mixer.init()
sound = mixer.Sound('alarm2.wav')
font = cv2.FONT_ITALIC
frame_no=0
matrix = [[0] * 50 for i in range(10)]
categories = ['danger', 'drowsy', 'normal']
karar= zerolist(50)
f=0
it=0
mat_pred=3
coffee = cv2.imread('coffee.jpg')
coffee = cv2.resize(coffee, (100,100))
drowsy_counter = 0

print('Running inference for {}... '.format(VIDEO_PATHS), end='')

from keras.models import load_model
model = load_model('models/karar_model_genis.h5')


video = cv2.VideoCapture(VIDEO_PATHS)
while(video.isOpened()):
    
    
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    
    if not ret:
        break
    frame = cv2.resize(frame, frame_size)
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
    cv2.rectangle(frame_with_detections, (0, 0), (210, 55), (0,0,0), -1)
    
    frame_no+=1 
    for r in range(0,8):
        if r in detection_names:
            matrix[frame_no][(r*6):(r*6+3)] = [1,1,1]
    matrix.append(zerolist(50))
        

    tekrar = 50    # x framede bir kontrol
    threshold = tekrar*0.7
    f = (f + 1) % tekrar # Update the iteration
    
    
    if f == 0:
        it+=1
        
        matris=np.copy(matrix)
        matris=np.delete(matris,np.s_[len(matris)-11:len(matris)-1],axis=0)
        matris = matris[(it-1)*tekrar:][:]
        
        sonuc = matris.sum(axis=0)
   
        karar = [1 if sonuc > round(threshold) else 0 for sonuc in sonuc]
        if (sonuc[4*6]>tekrar*0.25):                                         
            karar[(4*6):(4*7-1)]=[1,1,1]
            
        elif (sonuc[5*6]>tekrar*0.2):
            karar[(5*6):(5*7-1)]=[1,1,1]
            
        elif (sonuc[6*6]>tekrar*0.2):
            karar[(6*6):(6*7-1)]=[1,1,1]

        karar = kararver(karar)

        matris = matris.reshape(50,50,-1)
        matris = np.expand_dims(matris,axis=0)
        mat_pred = model.predict_classes(matris)
        yuzde = model.predict(matris)
        
        
        matris=0
    
        
        print("\n___%d___" %(it))
        # print("sonuc: ",sonuc)
        print("karar: ",karar)
        print("prediction: ",categories[mat_pred[0]])
        
        if mat_pred == 1:
            drowsy_counter = drowsy_counter + 1
        # print(yuzde)
    
    # if not any(sonuc):
    #     cv2.putText(frame,'no driver', (40, 35), font, 1, (255, 255, 255), 2, cv2.LINE_4)  # hicbir sey tespit edilemezse 'surucu yok' kararı ver

    if mat_pred==3:    
        cv2.putText(frame_with_detections,'welcome', (40, 35), font, 1, (255, 255, 255), 2, cv2.LINE_4)
    elif mat_pred==0:
        cv2.putText(frame_with_detections,'danger', (10, 35), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.putText(frame_with_detections,'%'+str(int(round(yuzde[0][0]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
        try:
            sound.play()    # danger durumunda alarm cal
        except:  
            pass    
    elif mat_pred==1:
        cv2.putText(frame_with_detections,'drowsy', (10, 35), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame_with_detections,'%'+str(int(round(yuzde[0][1]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
    elif mat_pred==2:
        cv2.putText(frame_with_detections,'normal', (10, 35), font, 1, (0, 255, 0), 2, cv2.LINE_4)  
        cv2.putText(frame_with_detections,'%'+str(int(round(yuzde[0][2]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
        
    if drowsy_counter >= 1:
        frame_with_detections[55:155,0:100] = coffee
    
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
          skip_scores=True,
          min_score_thresh=MIN_CONF_THRESH,
          agnostic_mode=False)
    # out.write(frame_with_detections)
    cv2.imshow('Monitoring System', frame_with_detections)
    
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print("Done")
print("frame_no: ", frame_no)