# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import category, image, number, boolean
import numpy as np
import tensorflow as tf
from PIL import Image
# from example_model import ExampleModel

import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.
@runway.setup()
def setup():
    msg = '[SETUP] Ran with options: network = {}'
    
    model = models.InceptionV1()
    model.load_graphdef()
    return model

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html

input_options = {
  'layer': category(choices=["conv2d0 (max:64)", "maxpool0 (max:64)", "conv2d1 (max:64)", "conv2d2 (max:192)", "maxpool1 (max:192)", "mixed3a (max:256)", "mixed3b (max:480)", "maxpool4 (max:480)", "mixed4a (max:508)", "mixed4b (max:512)", "mixed4c (max:512)", "mixed4d (max:528)", "mixed4e (max:832)", "maxpool10 (max:832)", "mixed5a (max:832)", "mixed5b (max:1024)"], default="mixed5b (max:1024)", description='choose layer of network to visualize'),
  'neuron': number(default=0, min=0, max=1023, step=1, description='Neuron ID'),
  'size': number(default=128, min=128, max=1024, step=128, description='Image Size'),
  'transforms': boolean(default=False, description='Vary size of visualization'),
  'transform_min': number(default=0.3, min=0.0, max=1.0, step=.1, description='Minimum scaling amount'),
  'transform_max': number(default=0.5, min=0.0, max=1.0, step=.1, description='Maximum scaling amount')
}

@runway.command(name='generate',
                inputs=input_options,
                outputs={ 'image': image() },
                description='Use Lucid to visualize the layers and neurons of a specific ML network.')
def generate(model, args):
    print('[GENERATE] Ran with layer {} and neuron {}'.format(args['layer'], args['neuron']))

    layer_id = args['layer'].split(' ')[0]
    layer_neuron = '{}:{}'.format(layer_id, args['neuron'])

    s = int(args['size'])
    min_scale = args['transform_min']
    max_scale = args['transform_max']
    scale_offset = (max_scale - min_scale) * 10

    # https://github.com/tensorflow/lucid/issues/148
    with tf.Graph().as_default() as graph, tf.Session() as sess:
  
        t_img = param.image(s)
        crop_W = int(s/2)
        t_offset = tf.random.uniform((2,), 0, s - crop_W, dtype="int32")
        t_img_crop = t_img[:, t_offset[0]:t_offset[0]+crop_W, t_offset[1]:t_offset[1]+crop_W]
        
        
        if(args['transforms']):
            transforms=[transform.jitter(2), 
                transform.random_scale([min_scale + n/10. for n in range(scale_offset)]),
                transform.random_rotate(range(-10,11)),
                transform.jitter(2)]
      
            T = render.make_vis_T(model, layer_neuron, t_img_crop, transforms=transforms)
        else:
            T = render.make_vis_T(model, layer_neuron, t_img_crop)

        tf.initialize_all_variables().run()

        for i in range(1024):
            T("vis_op").run()
          
        img = t_img.eval()[0]

    # https://github.com/tensorflow/lucid/issues/108
    # img = render.render_vis(model, layer_neuron)[-1][0]
    img = Image.fromarray(np.uint8(img*255))
    return {
        'image': img
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)
