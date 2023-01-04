#import mtcnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import cv2
from mtcnn import MTCNN
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle
from sklearn.model_selection import train_test_split



path = r'smile_dataset\*\*'
data  = []
label = []
for i, path in enumerate(glob.glob(path)):
    print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))/255.0
    #img = img/255.0
    ###### Detect Face
    Detector = MTCNN()
    try:
        face_index = Detector.detect_faces(img)[0]
        x, y, w, h = face_index['box']
        #cv2.rectangle(img, (x, y), (x+w, y+h), (255 , 0, 0), 2)

        ##### Detect elements
        kp= face_index['keypoints']
        for key, value in kp.items():
            cv2.circle(img, value, 3, (0,0,255),-1)

        img = img[x:x+w , y:y+h]
        data.append(img)
        labels = path.split("\\")[-2]
        label.append(labels)
        if i % 100 == 0:
            print(f"[Info] the {i}th prpcessed!")   
    except:
        pass


data = np.array(data, dtype=object)
label = np.array(label)
#print(label)

with open('data', 'wb') as config_dictionary_file:
  pickle.dump(data, config_dictionary_file)

with open('label', 'wb') as config_dictionary_file:
  pickle.dump(label, config_dictionary_file)

#data = np.asarray(data).astype(np.float32)
#label = np.asarray(label).astype(np.float32)
"""data = tf.convert_to_tensor(data, dtype=tf.int64)
label = tf.convert_to_tensor(label, dtype=tf.int64)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)
# np.savetxt('Smile_data.txt',data)
# np.savetxt('Smile_label.txt',label)

#cv2.imshow("Image", img)
#cv2.waitKey()
#cv2.destroyAllWindows()

CNN_net = models.Sequential([
                            layers.Conv2D(32, (3,3), activation = "relu", input_shape = (32,32,3)),
                            layers.MaxPool2D(),
                            layers.BatchNormalization(),
                            layers.Conv2D(64, (3,3), activation = "relu"),
                            layers.MaxPool2D(),
                            layers.BatchNormalization(),
                            layers.Flatten(),
                            layers.Dense(2, activation = "softmax")
                            ])
opt = SGD(learning_rate = 0.01, decay = 0.00025)
CNN_net.compile(
                optimizer = opt,
                metrics = ['accuracy'],
                loss = 'categorical_crossentropy'
                )
CNN_net.fit(
            X_train,
            y_train,
            batch_size = 32,
            validation_data = (X_test, y_test),
            epochs = 20
            )"""














