
import keras
from keras.models import Model
from keras.optimizers import adam_v2, gradient_descent_v2
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np


BATCH_SIZE = 20
RANDOM_SEED = 123

train_data_dir = "/Users/maxim/Downloads/Thesis code/Thesis data/train" 
test_data_dir = "/Users/maxim/Downloads/Thesis code/Thesis data/test"

train_generator = ImageDataGenerator(rotation_range=90, 
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)



traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=RANDOM_SEED)                    

validgen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=RANDOM_SEED)

testgen = test_generator.flow_from_directory(test_data_dir,
                                             target_size=(224, 224),
                                             class_mode='categorical',
                                             batch_size=10,
                                             shuffle=False)


def init_model(shape, optimizer, tunable=0):

        # PRELOAD IMAGENET!
    bottom = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=shape)

    # make n amounts of last conv layers trainable
    if tunable > 0:
        for layer in bottom.layers[:-tunable]:
            layer.trainable = False
    else:
        for layer in bottom.layers:
            layer.trainable = False

    top = Flatten(name="flatten")(bottom.output)
    top = Dense(4096, activation='relu')(top)
    top = Dense(1072, activation='relu')(top)
    top = Dropout(0.2)(top)
    output_layer = Dense(2, activation='sigmoid')(top)
    model = Model(inputs=bottom.input, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
    

shape = (224, 224, 3)
opt1 = adam_v2.Adam(learning_rate=0.001) # here for early testing
opt2 = gradient_descent_v2.SGD(learning_rate=0.001) #no momentum
n_epoch = 50

model = init_model(shape, opt2, tunable=0)

checkpoint = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)


early_stopping = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


training = model.fit(traingen,
                            batch_size=BATCH_SIZE,
                            epochs=n_epoch,
                            validation_data=validgen,
                            callbacks=[early_stopping,checkpoint],
                            verbose=1)


#uncomment section below for testing using selected hyperparameters

'''model.load_weights('tl_model_v1.weights.best.hdf5')

true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

preds = model.predict(testgen)
pred_classes = np.argmax(preds, axis=1)
#choose class with highest assigned probability from its own independent sigmoid function.


from sklearn.metrics import accuracy_score,auc,confusion_matrix
from sklearn import metrics

acc = accuracy_score(true_classes, pred_classes)
print("Testing Accuracy: {:.2f}%".format(acc * 100))
print(confusion_matrix(true_classes,pred_classes))
fpr, tpr, thresholds = metrics.roc_curve(true_classes,pred_classes, pos_label=1)
print(metrics.auc(fpr, tpr))'''


