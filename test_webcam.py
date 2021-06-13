import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow as tf
import cv2
import numpy as np
import warnings
from pygame import mixer
import time
from object_detection.utils import label_map_util
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
from some_functions import kararver
from some_functions import zerolist

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# kayıtlı model
PATH_TO_MODEL_DIR = 'exported-models_bitirme_yeni/my_mobilenet_model'
# etiketler
PATH_TO_LABELS = 'exported-models_bitirme_yeni/my_mobilenet_model/saved_model/label_map.pbtxt'
# nesne tanıma icin minimum threshold
MIN_CONF_THRESH = 0.5
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings


mixer.init()
sound = mixer.Sound('alarm2.wav')
font = cv2.FONT_ITALIC
mat_pred=3  # predict baslamadan once 'welcome' sabiti
frame_no=0  
categories = ['danger', 'drowsy', 'normal']
nodriver=0
matrix = [[0] * 50 for i in range(10)]  # baslangıc icin bos matrix
karar= zerolist(50)
f=0
it=0
coffee = cv2.imread('coffee.jpg')
coffee = cv2.resize(coffee, (100,100))
drowsy_counter = 0

print('Running inference for Webcam', end='')

# Initialize Webcam
videostream = cv2.VideoCapture(0)
ret = videostream.set(3,1280)
ret = videostream.set(4,720)


from keras.models import load_model
model_cnn = load_model('models/karar_model_genis.h5')   # karar icin cnn modelini cagır



while True:
    
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = videostream.read()
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_expanded = np.expand_dims(frame_rgb, axis=0)
    imH, imW, _ = frame.shape

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

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    cv2.rectangle(frame, (0, 0), (210, 55), (0,0,0), -1)    # sol uste siyah dikdortgen
  
    # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS
    
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    names = np.squeeze(detections['detection_classes'])
    detection_names = names[scores > MIN_CONF_THRESH]   # nesne tanıma minimum thresholdunu gecen classlar
    count = 0
    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            #increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Draw label
            object_name = category_index[int(names[i])]['name'] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            font = cv2.FONT_ITALIC
            

    frame_no+=1     # her frame icin frame sayısını say
    
    for r in range(0,8):    # 7 tane class var
        if r in detection_names:                # eger bu 7 classtan herhangi biri tespit edilirse# bos matrixin o sutununa 1 yaz
            matrix[frame_no][(r*6):(r*6+3)] = [1,1,1]         # bos matrixin o sutununa 1 yaz
    matrix.append(zerolist(50))                 # yeni frameler icin matrixe yeni satır ekle
    
    tekrar = 50    # x framede bir kontrol
    threshold = tekrar*0.7  # 50 frame'in en az 35'inde yapılan bir hareket gecerli sayılacak.
    f = (f + 1) % tekrar # her 50 framede bir kontrol edilmesi icin iterasyon tanımla
    
    
    if f == 0:      
        it+=1
        
        matris=np.copy(matrix)
        matris=np.delete(matris,np.s_[len(matris)-11:len(matris)-1],axis=0)   # basta eklenen bos matrixi sil
        matris = matris[(it-1)*tekrar:][:]      # matrisin sadece o anki iterasyonundaki 50 satırını al
        
        sonuc = matris.sum(axis=0)      # bu 50 satırı alt alta topla ve bir sonuc elde et
        
        # eger bu sonuctaki degerler, thresholdu (35'i) geciyorsa karar listesinin o sutunundaki degerini 1 yap
        karar = [1 if sonuc > round(threshold) else 0 for sonuc in sonuc]
        
        # esneme, gunes gozlugu ve kafayı asagı egme hareketleri icin thresholdu daha az tut.
        # bu hareketlerden az tespit edilse bile gecerli sayılsın
        if (sonuc[4*6]>tekrar*0.25):                                         
            karar[(4*6):(4*7-1)]=[1,1,1]
            
        elif (sonuc[5*6]>tekrar*0.2):
            karar[(5*6):(5*7-1)]=[1,1,1]
            
        elif (sonuc[6*6]>tekrar*0.2):
            karar[(6*6):(6*7-1)]=[1,1,1]
        
        # if not any(sonuc):
        #     nodriver=1
            
        karar = kararver(karar)     # 50 elemanlı diziyi 7 elemanlı diziye cevir.

        
        matris = matris.reshape(50,50,-1)
        matris = np.expand_dims(matris,axis=0)
        mat_pred = model_cnn.predict_classes(matris)    # durum tespiti icin cnn modelinde predict et
        yuzde = model_cnn.predict(matris)
                
        matris=0    # bir sonraki 50 frame icin matrisi sıfırla
    
        print("\n___%d___" %(it))
        # print("sonuc: ",sonuc)
        print("karar: ",karar)
        print("prediction: ",categories[mat_pred[0]])
        
        if mat_pred == 1:
            drowsy_counter = drowsy_counter + 1
    

    # if nodriver==1:
    #     cv2.putText(frame,'no driver', (40, 35), font, 1, (255, 255, 255), 2, cv2.LINE_4)  # hicbir sey tespit edilemezse 'surucu yok' kararı ver
    # nodriver=0 
    if mat_pred==3:    
        cv2.putText(frame,'welcome', (40, 35), font, 1, (255, 255, 255), 2, cv2.LINE_4)
    elif mat_pred==0:
        cv2.putText(frame,'danger', (10, 35), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.putText(frame,'%'+str(int(round(yuzde[0][0]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
        try:
            sound.play()    # danger durumunda alarm cal
        except:  
            pass
    elif mat_pred==1:
        cv2.putText(frame,'drowsy', (10, 35), font, 1, (0, 255, 255), 2, cv2.LINE_4)
        cv2.putText(frame,'%'+str(int(round(yuzde[0][1]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
        
    elif mat_pred==2:
        cv2.putText(frame,'normal', (10, 35), font, 1, (0, 255, 0), 2, cv2.LINE_4)  
        cv2.putText(frame,'%'+str(int(round(yuzde[0][2]*100))),(122, 35), font, 1,(255,255,255),2,cv2.LINE_4)
    
    if drowsy_counter >= 5:
        frame[55:155,0:100] = coffee

        
    
    cv2.imshow('Monitoring System', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print("Done")
print("frame_no: ", frame_no)
print("it: ",it)










