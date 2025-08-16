import tensorflow as tf
import random
import matplotlib.pyplot as plt

# Load model
model = tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet',
    input_shape=(96,96,3)
)

# Sub-model for specific layer
def get_subModel(layer_name):
    return tf.keras.models.Model(
        model.input,
        model.get_layer(layer_name).output
    )

# Create random image
def create_image():
    return tf.random.uniform((96,96,3), minval=-0.5, maxval=0.5)

# Normalize and plot image
def plot_image(image, title='random'):
    image = image - tf.reduce_min(image)
    image = image / tf.reduce_max(image)
    plt.imshow(image)
    plt.xticks([]); plt.yticks([])
    plt.title(title)
    plt.show()

# Visualize filter
def visualize_filter(layer_name, filter_index=None, iters=50):
    submodel = get_subModel(layer_name)
    num_filters = submodel.output.shape[-1]
    if filter_index is None:
        filter_index = random.randint(0, num_filters - 1)

    image = create_image()
    for i in range(iters):
        with tf.GradientTape() as tape:
            tape.watch(image)
            out = submodel(tf.expand_dims(image, 0))[:, :, :, filter_index]
            loss = tf.reduce_mean(out)
        grads = tape.gradient(loss, image)
        grads = tf.math.l2_normalize(grads)
        image += grads * 10
    plot_image(image, f'{layer_name}, {filter_index}')

# Example usage
layer_name = 'block3_conv3'
visualize_filter(layer_name, iters=150)
