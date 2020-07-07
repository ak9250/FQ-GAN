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
    
#decorater wraps around the function, its a function that operates on the function, slidely modifies the original function
#command is a function and you pass in the translate function 
# its a way of transforming a function into something else
#takes in a specific input and returns a output to display a image
@runway.command('translate', inputs={'image': runway.image}, outputs={'image': runway.image})

def translate(gan, inputs):
    original_size = inputs['image'].size
    img = inputs['image'].resize((256, 256))
    output = gan.generate(img)
    # the range of data is between -1 and 1, normally clip it between 0 and 1
    output = np.clip(output, -1, 1)
    #noramalization by adding 1 and multiplying 255 and dividing 2 
    output = ((output + 1.0) * 255 / 2.0)
    #turns it into a unsigned 8 bit integer, common image format
    output = output.astype(np.uint8)
    #turns the array back into a image
    return Image.fromarray(output).resize(original_size)

if __name__ == '__main__':
    #calls both command and setup, similar to flask, its like a version of flask thats handles in the inputs for you
    # flask is more basic, handles endpoints
    # runway basically creates a webpage and the desktop app is the container for that webpage
    # presumably the runway desktop app is a electron app, like a chrome 
    # uses html and javascript to display content
    # its a quick and easy way to create desktop appplications, instead of tkinter to create a desktop app you use html and javascript
    runway.run(port=8889)