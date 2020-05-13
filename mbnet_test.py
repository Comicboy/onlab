import tensorflow as tf
from timeit import default_timer as timer

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import preprocess_input
from keras.applications.mobilenet import decode_predictions

model = tf.keras.models.load_model('mbnet75.h5')

model.summary()

image = load_img(r"C:\Users\peisz\Downloads\muggcomp.jpg", target_size=(224, 224))


# A kép pixeleinek konverziója egy numpy tömbbe
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))


# Kép előfeldolgozása a Keras belső függvényével
image = preprocess_input(image)

# Mérés elindítása, majd preikció indítása
start = timer()
for i in range(1000):
    yhat = model.predict(image)

end = timer()

# Predikció dekódolása egy cimkére
label = decode_predictions(yhat)

# Legmagasabb valószínűségű predikció kiválasztása
label = label[0][0]

# A klasszifikáció eredményének, valamint az egy predikcióhoz szükséges átlagos idő kiírása
print('%s (%.2f%%)' % (label[1], label[2] * 100))
print((end-start)/1000)
