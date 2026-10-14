The project first has a model build from scratch which has the model definition as --

model=models.Sequential([
    layers.Input((224,224,3)),
    augmentation,
    layers.Conv2D(32,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(64,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(128,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.Conv2D(256,3,activation=activation,padding='same'),
    layers.MaxPooling2D(2,padding='same'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128,activation=activation),
    layers.Dense(64,activation=activation),
    layers.Dropout(0.5),
    layers.Dense(2,activation='softmax',dtype='float32')
])

This model due to the datasets high precentage of non cancer images to cancer images fails to accurately predict the cancer images in the testing phase.The model has a testing accuracy of -- 71.14%

Then I used the pre trained model of Efficient B3 and then used transfer learning to train this new model to the Breast Cancer Dataset.To protect the model from failing on slightly different or angled images, I used data augmentation during the training phase of the model so it learns on different kinds on images and not just on the particular images in the training dataset.The data augmentation was done by --

model = tf.keras.Sequential([
    layers.Input((224,224,3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.4),
    layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

This model performs well on the testing dataset with a testing accuracy of -- 92.62%
