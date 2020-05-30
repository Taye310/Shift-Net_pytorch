import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import ntpath
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        t1 = time.time()
        model.set_input(data)
        model.test()
        t2 = time.time()
        print(t2-t1)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        pred_depth = visuals['fake_B'][0][0].cpu().float().numpy()

        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        # save_path = '/home/zhangtianyi/ShareFolder/data/hmd_masked/test/shift-net_depth_result/' + name + '.npy'
        save_path = '/home/zhangtianyi/github/hmd/eval/eval_data/syn_set/pred_depth/' + name
        print(pred_depth.shape,save_path)
        np.save(save_path  + '.npy', pred_depth)
        plt.imsave(save_path + '.png', pred_depth)

        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()
