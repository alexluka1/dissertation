# Example of using weight and bias initiasation
kernel_initialiser = initializers.RandomNormal(seed=seed_value)
bias_initialiser =  initializers.Zeros()
model.add(layers.Dense(840, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))

lr = 0.001 # Unless specified


# Base model --------------------------------------------------
model = keras.Sequential()
# Even if grey scale, tf expects channel input
input_layer = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize[0],imageSize[1],1))
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='softmax'))

# --------------------------------------------------

model = keras.Sequential()
# Even if grey scale, tf expects channel input
input_layer = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5, seed=seed_value))
model.add(layers.Dense(64, kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))




# Base model tuned --------------------------------------------------
model = keras.Sequential()
input_layer = layers.Conv2D(80, (3, 3), activation='relu', input_shape=(imageSize[0],imageSize[1],1))
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(80, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(112, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(160, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='softmax'))

model.summary()

# --------------------------------------------------

model = keras.Sequential()
input_layer = layers.Conv2D(80, (3, 3), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(80, (3, 3), activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(112, (3, 3), activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5,seed=seed_value))
model.add(layers.Dense(160, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Flatten())
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))

model.summary()

# model 1 --------------------------------------------------
lr = 0.001
model = keras.Sequential()
input_layer = layers.Conv2D(48, (2,2), activation='relu', input_shape=(imageSize[0],imageSize[1],1))
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

#              filters Mp Conv Kernel
convBlock(model, 88, (3,3),(1,1)) # 96 the wrong way around on model one

model.add(layers.Flatten())
model.add(layers.Dropout(0.6))
model.add(layers.Dense(248, activation='relu'))
model.add(layers.Dense(744, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# --------------------------------------------------

model = keras.Sequential()
input_layer = layers.Conv2D(48, (2,2), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

#              filters Mp Conv Kernel
convBlock(model, 88, (3,3),(1,1)) # 96 the wrong way around on model one

model.add(layers.Flatten())
model.add(layers.Dropout(0.6,seed=seed_value))
model.add(layers.Dense(248, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(744, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))



# model 2 -------------------------------------------------- do again with lr = 0.001
lr = 0.0001
model = keras.Sequential()
input_layer = layers.Conv2D(80, (3,3), activation='relu', input_shape=(imageSize[0],imageSize[1],1))
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

#              filters Mp Conv Kernel
convBlock(model, 56, (2,2),(1,1)) # 96 the wrong way around on model one
convBlock(model, 56, (2,2),(1,1)) 

model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(984, activation='relu'))
model.add(layers.Dense(520, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))


# --------------------------------------------------

model = keras.Sequential()
input_layer = layers.Conv2D(80, (3,3), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

#              filters Mp Conv Kernel
convBlock(model, 56, (2,2),(1,1)) # 96 the wrong way around on model one
convBlock(model, 56, (2,2),(1,1)) 

model.add(layers.Flatten())
model.add(layers.Dropout(0.2, seed=seed_value))
model.add(layers.Dense(984, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(520, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))


# model 3 --------------------------------------------------
lr = 0.0001
model = keras.Sequential()
input_layer = layers.Conv2D(80, (3,3), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)

#              filters Mp Conv Kernel
convBlock(model, 72, (2,2),(4,4)) # 96 the wrong way around on model one
convBlock(model, 72, (2,2),(4,4))
convBlock(model, 72, (2,2),(4,4))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5, seed=seed_value))
model.add(layers.Dense(840, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(1024, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))


# model 4 --------------------------------------------------
lr = 0.001
model = keras.Sequential()
input_layer = layers.Conv2D(72, (3,3), activation='relu', input_shape=(imageSize[0],imageSize[1],1), kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser)
input_layer._name = 'input' # Setting name of layer
model.add(input_layer)


model.add(layers.MaxPooling2D(2))
model.add(layers.Conv2D(112, (1,1), activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))

#        filters, mp, kernel
convBlock(model, 176, (4,4), (2,2))

model.add(layers.Flatten())
model.add(layers.Dropout(0.7, seed=seed_value))
model.add(layers.Dense(480, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(680, activation='relu', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))
model.add(layers.Dense(4, activation='softmax', kernel_initializer=kernel_initialiser, bias_initializer=bias_initialiser))



