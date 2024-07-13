import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
directory="C:/Users/adars/Downloads/homer_bart"
data=tf.keras.preprocessing.image_dataset_from_directory(directory,labels='inferred',label_mode='int',color_mode='rgb',batch_size=32, image_size=(64,64), shuffle='True')
model=keras.Sequential([layers.Flatten(input_shape=(64, 64, 3)),
                        layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01)),
                        layers.BatchNormalization(),
                        layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.01)),
                        layers.BatchNormalization(),
                        layers.Dense(10,activation = 'sigmoid',kernel_regularizer=regularizers.l2(0.01))
                        ])
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.0003),
              metrics=['accuracy'])
model.fit(data,epochs=30,verbose=2)
print("training done")
model.evaluate(data,batch_size=32,verbose=2)