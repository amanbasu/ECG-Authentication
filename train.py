from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Lambda, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

from data_generator import Siamese_Loader

def get_siamese_model(input_shape):
    """
    Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(32, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(Conv2D(256, (3,3), activation='relu', kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    
    model.add(Flatten())
    model.add(Dense(512, activation='sigmoid', kernel_regularizer=l2(1e-3)))
    
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    return siamese_net

def train_model(model, loader, weights_path="model.h5"):
    evaluate_every = 20     # interval for evaluating on one-shot tasks
    loss_every = 5          # interval for printing loss (iterations)
    batch_size = 32
    n_iter = 1000
    N_way = 5               # how many people for testing one-shot tasks?
    n_val = 50              # how many one-shot tasks to validate on?
    best = 0

    train_loss, train_acc = [], []
     
    print("Starting training process!")
    print("-------------------------------------")
    for i in range(1, n_iter):
        (inputs, targets) = loader.get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        train_loss.append(loss)
        print(loss)

        if i%loss_every == 0:
          print("iteration {}, training loss: {:.2f},".format(i,loss))

        if i%evaluate_every == 0:
          val_acc = loader.test_oneshot(model, N_way, n_val, verbose=True)
          if val_acc >= best:
              print("Current best: {0}, previous best: {1}\n".format(val_acc, best))
              print("Saving weights to: {0} \n".format(weights_path))
              model.save_weights(weights_path)
              best = val_acc

if __name__ == '__main__':
    model = get_siamese_model((144, 224, 1))
    optimizer = Adam(lr = 0.00006)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    loader = Siamese_Loader(path="path-to-data")
    train_model(model, loader, weights_path="model_ecg.h5")
