from .create_word_vector import create_word_vecs
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.optimizers import SGD
import seaborn as sns
from sklearn.metrics import log_loss


def build_model(architecture='mlp'):
    model = Sequential()
    if architecture == 'mlp':
        # Densely Connected Neural Network (Multi-Layer Perceptron)
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal', input_dim=300))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
    elif architecture == 'cnn':
        # 1-D Convolutional Neural Network
        inputs = Input(shape=(300,1))

        x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(inputs)

        #Cuts the size of the output in half, maxing over every 2 inputs
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='CNN')
    elif architecture == 'lstm':
        # LSTM network
        inputs = Input(shape=(300,1))

        x = Bidirectional(LSTM(64, return_sequences=True),
                          merge_mode='concat')(inputs)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='LSTM')
    else:
        print('Error: Model type not found.')
    return model


def run(articles, articles_test):
    X_train, X_test, y_train, y_test = create_word_vecs(articles, articles_test)
    print('X_train size: {}'.format(X_train.shape))
    print('X_test size: {}'.format(X_test.shape))
    print('y_train size: {}'.format(y_train.shape))
    print('y_test size: {}'.format(y_test.shape))

    model = build_model('mlp')


    if model.name == "CNN" or model.name == "LSTM":
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        print('Text train shape: ', X_train.shape)
        print('Text test shape: ', X_test.shape)

    model.summary()

    # Compile the model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

    # Define number of epochs
    epochs = 30

    # Fit the model to the training data
    estimator = model.fit(X_train, y_train,
                          validation_split=0.2,
                          epochs=epochs, batch_size=128, verbose=1)

    print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" %
          (100 * estimator.history['acc'][-1], 100 * estimator.history['val_acc'][-1]))
    # Plot model accuracy over epochs
    sns.reset_orig()  # Reset seaborn settings to get rid of black background
    plt.plot(estimator.history['acc'])
    plt.plot(estimator.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # Plot model loss over epochs
    plt.plot(estimator.history['loss'])
    plt.plot(estimator.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # Make predictions
    predicted_prob = model.predict(X_test)
    print(predicted_prob.shape)

    # Save submission file
    with open('submission.csv', 'w') as file_obj:
        file_obj.write('ID,JF,SPON,ZEIT\n')
        for pred in range(len(predicted_prob)):
            file_obj.write(
                str(pred + 1) + ',' + ','.join('{:.2f}'.format(s) for s in predicted_prob[pred].tolist()) + '\n')

            # Report log loss and score
            loss_sk = log_loss(y_test, predicted_prob)
            print('Log loss is: {}'.format(loss_sk))