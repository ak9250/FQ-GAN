import runway
import numpy as np
import tensorflow as tf
from UGATIT import UGATIT
from main import parse_args
from PIL import Image

g = tf.get_default_graph()
sess = tf.InteractiveSession(graph=g)

# setup up file loads the model and it calls ugatit load from latest 
@runway.setup(options={'checkpoint': runway.file(is_directory=True)})
def setup(opts):
    args = parse_args()
    args.dataset = 'portrait'
    gan = UGATIT(sess, args)
    gan.build_model()
    gan.load_from_latest(opts['checkpoint'])
    return gan
    
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})
def translate(gan, inputs):
    original_size = inputs['image'].size
    img = inputs['image'].resize((256, 256))
    output = gan.generate(img)
    output = np.clip(output, -1, 1)
    output = ((output + 1.0) * 255 / 2.0)
    output = output.astype(np.uint8)
    return Image.fromarray(output).resize(original_size)

if __name__ == '__main__':
    runway.run(port=8889)