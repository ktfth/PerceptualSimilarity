import os
import shutil

from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response

import models

from util import util

def hello_world(request):
    return Response('Hello World!')

def similarity(request):
    use_gpu = False
    spatial = False

    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, spatial=spatial)

    img1_filename = request.POST['img1'].filename
    img1_input_file = request.POST['img1'].file
    img1_file_path = os.path.join('/tmp', img1_filename)

    with open(img1_file_path, 'wb') as img1_output_file:
        shutil.copyfileobj(img1_input_file, img1_output_file)

    img1tensor = util.im2tensor(util.load_image(img1_file_path))

    img2_filename = request.POST['img2'].filename
    img2_input_file = request.POST['img2'].file
    img2_file_path = os.path.join('/tmp', img2_filename)

    with open(img2_file_path, 'wb') as img2_output_file:
        shutil.copyfileobj(img2_input_file, img2_output_file)

    img2tensor = util.im2tensor(util.load_image(img2_file_path))

    d = model.forward(img1tensor, img2tensor)

    return Response('Distance: %.3f' % d)

if __name__ == '__main__':
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        config.add_route('similarity', '/similarity')
        config.add_view(similarity, route_name='similarity')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', 6543, app)
    server.serve_forever()
