import os
import random
import time
import traceback
from types import SimpleNamespace

import cv2
import sys
from PySide6 import QtWidgets, QtGui, QtCore
import sys
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from PySide6.QtCore import Signal, QObject, QThreadPool, Slot, QRunnable
from PySide6.QtWidgets import QTextEdit, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QLabel
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import trange, tqdm

from conditioning import threshold_by, get_uc_and_c
from k_samplers import sampler_fn, make_inject_timing_fn
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser, VDenoiser, DiscreteVDDPMDenoiser, OpenAIDenoiser
from model_wrap import CFGDenoiserWithGrad

#from imwatermark import WatermarkEncoder

#from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS
from PIL.ImageQt import ImageQt

from resizeRight import resizeright, interp_methods

torch.set_grad_enabled(False)
import threading

import cv2
import numpy as np

def blend_images(img1, img2, num_frames=25):
    images = []
    if img1.size[0] != img2.size[0]:
        img1 = img1.resize(img2.size, resample=Image.Resampling.NEAREST)
    for i in range(num_frames):
        # Read the images as PIL images
        #img1 = np.array(img1)
        #img2 = np.array(img2)

        # Blend the two images together
        alpha = i / (num_frames - 1)

        img = Image.blend(img1, img2, alpha)

        # Convert the image to a numpy array and append to the list
        img_array = np.array(img)
        images.append(img_array)
    return images
def morph_images(image1, image2):
    # Convert the images to grayscale
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors of both images using ORB
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1_gray, None)
    kp2, des2 = orb.detectAndCompute(image2_gray, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(des1, des2)

    # Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Find the top 50 matches
    top_matches = matches[:256]

    # Find the coordinates of the top 50 matches
    points1 = np.zeros((len(top_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(top_matches), 2), dtype=np.float32)

    for i, match in enumerate(top_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    # Compute the affine transformation matrix using the top matches
    M, mask = cv2.estimateAffine2D(points1, points2)

    # Create the morph images
    morph_images = []
    for alpha in np.linspace(0, 1, num=128):
        # Warp image1 using the affine transformation matrix
        image1_warped = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))

        # Blend the two images using a weighted sum
        morph_image = cv2.addWeighted(image1_warped, 1 - alpha, image2, alpha, 0)
        morph_images.append(morph_image)

    return morph_images
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.half().to(device)
    #sampler = DDIMSampler(model)
    return model

def load_model_from_config(config, ckpt, verbose=False):
    config = OmegaConf.load(config)
    #print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        #print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        #print(u)
    model.half().cuda()
    model.eval()
    del sd
    del pl_sd

    return model




def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, lock=False, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.lock = lock
        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class OurSignals(QObject):
    updateimagesignal = Signal()
    webcamupdate = Signal()



class WebcamWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = OurSignals()
        self.signals.updateimagesignal.connect(self.update_image)
        self.signals.webcamupdate.connect(self.update_frame_func)
        # Set up the user interface
        self.capture_button = QtWidgets.QPushButton('Capture')
        self.capture_button.clicked.connect(self.capture_image)
        self.continous = QtWidgets.QPushButton('Continous')
        self.continous.clicked.connect(self.start_continuous_capture)
        self.webcam_dropdown = QtWidgets.QComboBox()
        self.webcam_dropdown.addItems(self.get_available_webcams())
        self.webcam_dropdown.currentIndexChanged.connect(self.start_webcam)
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(512, 512)
        #self.model = initialize_model("configs/stable-diffusion/v2-midas-inference.yaml",
        #                           "models/512-depth-ema.ckpt")
        self.model = load_model_from_config("configs/stable-diffusion/v1-inference.yaml",
                                   "models/v1-5-pruned-emaonly.ckpt")
        #self.sampler = DDIMSampler(self.model)

        self.prompt = QTextEdit()
        self.steps = QSpinBox()
        self.steps.setValue(20)
        self.steps.valueChanged.connect(self.make_sampler_schedule)

        self.strength = QDoubleSpinBox()
        self.strength.setValue(0.60)
        self.strength.setMaximum(1.00)
        self.strength.setMinimum(0.01)
        self.strength.setSingleStep(0.01)
        self.eta = QDoubleSpinBox()
        self.eta.setValue(0.0)
        self.eta.setMaximum(1.00)
        self.eta.setMinimum(0.00)
        self.eta.setSingleStep(0.01)

        self.rescalefactor = QDoubleSpinBox()
        self.rescalefactorlabel = QLabel("rescalefactor")
        self.rescalefactor.setValue(1.0)
        self.rescalefactor.setMinimum(1.0)
        self.rescalefactor.setMaximum(10.0)
        self.rescalefactor.setSingleStep(0.01)

        self.seed = QLineEdit()
        self.samplercombobox = QComboBox()
        self.samplercombobox.addItems(["klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral",
                                        "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde"])
        # Set up the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.strength)

        layout.addWidget(self.rescalefactorlabel)
        layout.addWidget(self.rescalefactor)
        layout.addWidget(self.eta)
        layout.addWidget(self.seed)
        layout.addWidget(self.samplercombobox)
        layout.addWidget(self.webcam_dropdown)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.continous)
        layout.addWidget(self.camera_label)
        self.setLayout(layout)
        self.threadpool = QThreadPool()
        # Start the webcam
        #webcamthread = Worker(self.start_webcam)
        #self.threadpool.start(webcamthread)
        self.start_webcam()
        self.morphed_images = []
        #Show the preview window
        self.image_label = QtWidgets.QLabel()
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        # Create a QDialog and set the layout to the layout containing the image label
        self.image_dialog = QtWidgets.QWidget()
        prevlayout = QtWidgets.QVBoxLayout()
        self.fullscreen_button = QtWidgets.QPushButton('Fullscreen')
        self.fullscreen_button.clicked.connect(self.show_fullscreen)
        prevlayout.addWidget(self.fullscreen_button)
        prevlayout.addWidget(self.image_label)
        self.image_dialog.setLayout(prevlayout)
        # Show the dialog
        self.image_dialog.show()
        model_type = "dpt_hybrid"
        #self.midas_trafo = AddMiDaS(model_type=model_type)



    def show_fullscreen(self):
        if self.image_dialog.isFullScreen():
            self.image_dialog.showNormal()
        else:
            self.image_dialog.showFullScreen()
    def get_available_webcams(self):
        """Get a list of available webcams."""
        webcams = []
        for i in range(10):
            capture = cv2.VideoCapture(i)
            if capture.isOpened():
                webcams.append(f'Webcam {i}')
            capture.release()
        return webcams

    def start_webcam(self):
        """Start the webcam and display the video feed."""
        self.capture = cv2.VideoCapture(self.webcam_dropdown.currentIndex())
        #if self.capture.isOpened():
        #    self.wtimer = QtCore.QTimer()
        #    self.wtimer.timeout.connect(self.update_frame)
        #    self.wtimer.start(8)

    def update_frame(self):
        """Update the camera preview label with the latest frame."""
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.webcamimage = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
        self.signals.webcamupdate.emit()

    @Slot()
    def update_frame_func(self):
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(self.webcamimage))
    def capture_image(self):
        """Capture an image from the webcam and display it in a new floating window."""
        self.run = False
        _, frame = self.capture.read()
        image = Image.fromarray(frame)

        # Call the paint function and get the resulting image
        result_image = self.img2img(image, "Monster in the shadow", 5, 1, 7.5, random.randint(0, 400000), 0.0, 0.6)

        # Create a QLabel to display the image
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(result_image)))
        self.image_label.setScaledContents(True)

    def start_continuous_capture(self):
        """Start the continuous capture in a separate thread."""
        self.run = True

        worker = Worker(self.continuous_capture)
        self.threadpool.start(worker)
        # Create a QTimer
        self.timer2 = QtCore.QTimer()

        # Set the timer to be a single-shot timer
        self.timer2.setSingleShot(True)

        # Set the timeout duration for the timer
        self.timer2.setInterval(2000)  # 1 second

        # Connect the timeout signal to a function
        self.timer2.timeout.connect(self.start_continous_capture_again)

        # Start the timer
        #self.timer2.start()

        self.timer = QtCore.QTimer()

        # Set the timer interval
        self.timer.setInterval(8)

        # Connect the timer's timeout signal to the update_image slot
        self.timer.timeout.connect(self.update_image_signal)
        self.index = 0
        # Start the timer
        self.timer.start()

        #self.capture_thread = threading.Thread(target=self.continuous_capture)
        #self.capture_thread.start()
    def start_continous_capture_again(self):
        worker2 = Worker(self.continuous_capture)
        self.threadpool.start(worker2)

    def make_sampler_schedule(self):
        steps = self.steps.value()
        eta = self.eta.value()
        self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)

    def continuous_capture(self, progress_callback=None):
        """Capture images from the webcam continuously."""
        # State for interpolating between diffusion steps
        result_images = []
        self.images = []
        #steps = self.steps.value()
        self.index = 0
        #self.make_sampler_schedule()
        self.lastinit = None
        #self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
        #self.iterator_thread = threading.Thread(target=self.iterator)
        #elf.iterator_thread.start()
        self.seedint = 0

        model_wrap = CompVisDenoiser(self.model, quantize=False)

        loss_fns_scales = [
            [None, 0.0],
            [None, 0.0],
            [None, 0.0],
            [None, 0.0],
            [None, 0.0],
            [None, 0.0],
            [None, 0.0],
            [None, 0.0]
        ]
        clamp_fn = threshold_by(threshold=0, threshold_type='dynamic',
                                clamp_schedule=[0])
        grad_inject_timing_fn = make_inject_timing_fn(None, model_wrap, 10)
        self.cfg_model = CFGDenoiserWithGrad(model_wrap,
                                        loss_fns_scales,
                                        clamp_fn,
                                        None,
                                        None,
                                        True,
                                        decode_method=None,
                                        grad_inject_timing_fn=grad_inject_timing_fn,
                                        grad_consolidate_fn=None,
                                        verbose=False)


        while self.run == True:
            _, frame = self.capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(frame)

            # Call the predict function and get the resulting image
            prompt = self.prompt.toPlainText()
            steps = self.steps.value()
            strength = self.strength.value()
            self.seedint = self.seed.text() if self.seed.text() != '' else self.seedint
            self.seedint = int(self.seedint) + 1 if self.seedint != '' else random.randint(0, 4000000)
            eta = self.eta.value()
            #factor = 1.10
            #image = image.resize((int(image.size[0] / factor), int(image.size[1] / factor)),
            #                                 resample=Image.Resampling.NEAREST)

            #result_image = self.img2img(small_image, prompt,
            #                            3, 1, 7.5, self.seedint, eta, 0.40)

            result_image = self.img2img(frame, prompt,
                                        steps, 1, 12.5, self.seedint, eta, strength)

            # Store the result image in a list
            #result_images.append(result_image[1])
            self.images.append(result_image)
            self.morphed_images = []
            if len(self.images) > 1:
                self.index = 0
                self.morphed_images = blend_images(self.images[len(self.images) - 2], self.images[len(self.images) - 1])
            # If there are two result images, interpolate between them using Image.blend
            """if len(result_images) == 2:
                # Interpolate between the result images using Image.blend
                #images = []
                for i in range(1, 51):
                    current_image = result_images[0]
                    next_image = result_images[1]
                    interpolated_image = Image.blend(current_image, next_image, i / 50)
                    #images.append(interpolated_image)
                    self.images.append(interpolated_image)
                #self.timer = QtCore.QTimer()

                # Set the timer interval
                #self.timer.setInterval(8)

                # Connect the timer's timeout signal to the update_image slot
                #self.timer.timeout.connect(self.update_image_signal)
                # Start the timer
                #self.timer.start()
                # Display the last interpolated image on the QLabel
                    #self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(interpolated_image)))
                    #self.image_label.setScaledContents(True)

                # Set the second result image as the first element of the list
                result_images = [result_images[1]]"""
            if self.run == False:
                break

    def iterator(self, progress_callback=None):
        self.timer = QtCore.QTimer()

        # Set the timer interval
        self.timer.setInterval(8)

        # Connect the timer's timeout signal to the update_image slot
        self.timer.timeout.connect(self.update_image_signal)
        self.index = 0
        # Start the timer
        self.timer.start()
    def update_image_signal(self):
        self.signals.updateimagesignal.emit()
    @Slot()
    def update_image(self):
        ##print("updateinSS")
        # Increment the index
        self.index += 1
        donotdraw = None
        if self.index >= len(self.morphed_images) - 1:
            self.index = len(self.morphed_images) - 1
            donotdraw = True
        if self.morphed_images != []:
            if donotdraw is not True:
                self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.morphed_images[self.index]))))
                self.image_label.setScaledContents(True)

        """
        # If the index is out of range, reset it to 0
        if self.index >= len(self.images):
            if len(self.images) > 0:
                self.index = len(self.images) - 1
                return
                images = []
                for i in range(1, 51):
                    current_image = self.images[len(self.images) - 1]
                    next_image = self.images[0]
                    interpolated_image = Image.blend(current_image, next_image, i / 50)
                    images.append(interpolated_image)
                    ##print("0")
                    self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(interpolated_image)))
                    #self.images.append(interpolated_image)

            #self.index = 0
            #self.timer.stop()
            #self.timer.timeout.disconnect()

        # Set the pixmap to the next image
        if len(self.images) > 1:
            morphed_images = morph_images(self.images[self.index - 1], self.images[self.index])
            for i in morphed_images:
                self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(i))))
                #Image.fromarray(i).save(f"{random.randint(0,9999)}t.png")
            self.image_label.setScaledContents(True)

        #self.iterator()"""

    def img2img(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        #factor = 1.25
        #image = input_image.resize((int(input_image.size[0] / factor), int(input_image.size[1] / factor)), resample=Image.Resampling.NEAREST)

        #if self.lastinit is not None:
        #    #print(self.lastinit, input_image)

        #    self.lastinit = self.lastinit.resize((int(input_image.size[0]), (input_image.size[1])), resample=Image.Resampling.LANCZOS)
        #    input_image = Image.blend(self.lastinit, input_image, 0.85)
        #model = self.model

        #image = np.array(input_image).astype(np.float32) / 255.0
        print(input_image.shape)
        image = input_image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.*image - 1.
        image = image.half().to("cuda")

        data = [1 * [prompt]]
        t_enc = int(strength * steps)
        device = "cuda"
        seed_everything(seed)
        factor = self.rescalefactor.value()
        with torch.no_grad():
            #torch.backends.cudnn.benchmark = True
            with autocast("cuda"):
                with self.model.ema_scope():
                    init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(image))
                    init_latent = resizeright.resize(init_latent, scale_factors=None,
                                                 out_shape=[init_latent.shape[0], init_latent.shape[1], int(init_latent.shape[2] // factor),
                                                            int(init_latent.shape[3] // factor)],
                                                 interp_method=interp_methods.lanczos3, support_sz=None,
                                                 antialiasing=True, by_convs=True, scale_tolerance=None,
                                                 max_numerator=10, pad_mode='reflect')

                    #tic = time.time()
                    #all_samples = list()
                    for n in trange(1, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            #uc = None
                            #if scale != 1.0:
                            #    uc = self.model.get_learned_conditioning(1 * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            #c = self.model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            #z_enc = self.sampler.stochastic_encode(init_latent,
                            #                                  torch.tensor([t_enc] * 1).to(device))
                            # decode it
                            #samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                            #                         unconditional_conditioning=uc, )
                            args = SimpleNamespace()
                            args.use_init = True
                            args.scale = scale
                            args.sampler = self.samplercombobox.currentText()
                            args.n_samples = 1
                            args.C = 4
                            args.W = int(input_image.shape[1] / factor)
                            args.H = int(input_image.shape[0] / factor)
                            args.f = 8
                            args.steps = steps
                            args.log_weighted_subprompts = False
                            args.normalize_prompt_weights = True
                            uc, c = get_uc_and_c(prompts, self.model, args, 0)
                            samples = sampler_fn(
                                c=c,
                                uc=uc,
                                args=args,
                                model_wrap=self.cfg_model,
                                init_latent=init_latent,
                                t_enc=t_enc,
                                device="cuda",
                                cb=None,
                                verbose=False)

                            x_samples = self.model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            #if not opt.skip_save:
                            #    for x_sample in x_samples:
                            #        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            #        Image.fromarray(x_sample.astype(np.uint8)).save(
                            #            os.path.join(sample_path, f"{base_count:05}.png"))
                            #        base_count += 1
                            #all_samples.append(x_samples)

                    #if not opt.skip_grid:
                    #    # additionally, save as grid
                    #    grid = torch.stack(all_samples, 0)
                    #    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    #    grid = make_grid(grid, nrow=n_rows)

                        # to image
                    #   grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    #    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    #    grid_count += 1

                    #toc = time.time()
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                    #self.lastinit = image
                    return image

    def predict(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        do_full_sample = strength == 1.
        t_enc = min(int(strength * steps), steps-1)
        width, height = input_image.size
        # Calculate the width and height of each quadrant
        quad_width = width // 2
        quad_height = height // 2
        
        overlap = 0
        
        # Create the four quadrants
        #quad1 = input_image.crop((-overlap, -overlap, quad_width + overlap, quad_height + overlap))
        #quad2 = input_image.crop((quad_width - overlap, -overlap, width + overlap, quad_height + overlap))
        #quad3 = input_image.crop((-overlap, quad_height - overlap, quad_width + overlap, height + overlap))
        #quad4 = input_image.crop((quad_width - overlap, quad_height - overlap, width + overlap, height + overlap))
        #quads = [quad1, quad2, quad3, quad4]
        #resquads = []
        #for i in quads:

        result = self.paint(
            sampler=self.sampler,
            model=self.sampler.model,
            image=input_image,
            image_quad=input_image,
            prompt=prompt,
            t_enc=t_enc,
            seed=seed,
            scale=scale,
            num_samples=num_samples,
            callback=None,
            do_full_sample=do_full_sample
            )
        #    resquads.append(result)
        # Create a blank image with the same size as the input image
        #resimage = Image.new('RGBA', (width, height))
        #quad_width = 10
        # Paste the quadrants into the blank image
        #resimage.paste(resquads[0], (0, 0))
        #resimage.paste(resquads[1], (quad_width, 0))
        #resimage.paste(resquads[2], (0, quad_height))
        #resimage.paste(resquads[3], (quad_width, quad_height))

        return result


    def paint(self, sampler, model, image, image_quad, prompt, t_enc, seed, scale, num_samples=1, callback=None,
              do_full_sample=False):
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        #model = sampler.model
        seed_everything(seed)

        ##print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        #wm = "SDV2"
        #wm_encoder = WatermarkEncoder()
        #wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        with torch.no_grad(),\
                torch.autocast("cuda"):
            batch = self.make_batch_sd(
                image, txt=prompt, device=device, num_samples=num_samples)
            z = model.get_first_stage_encoding(model.encode_first_stage(
                batch[model.first_stage_key]))


            #image = image.resize(size=(int(image.size[0] / 2), int(image.size[1] / 2)), resample=Image.Resampling.NEAREST)

            #batch2 = self.make_batch_sd(
            #    image_quad, txt=prompt, device=device, num_samples=num_samples)
            #z2 = model.get_first_stage_encoding(model.encode_first_stage(
            #    batch[model.first_stage_key]))

            # move to latent space
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck]
                cc = model.depth_model(cc)
                #depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                #                                                                               keepdim=True)
                #display_depth = (cc - depth_min) / (depth_max - depth_min)
                #depth_image = Image.fromarray(
                #    (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
                cc = torch.nn.functional.interpolate(
                    cc,
                    size=z.shape[2:],
                    mode="bicubic",
                    align_corners=False,
                )
                depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                               keepdim=True)
                cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            #if not do_full_sample:
            #    # encode (scaled latent)
            #    z_enc = sampler.stochastic_encode(
            #        z, torch.tensor([t_enc] * num_samples).to(model.device))
            #else:
            z_enc = torch.randn_like(z)
            # decode it

            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc_full, callback=callback)
            x_samples_ddim = model.decode_first_stage(samples)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
            image = [Image.fromarray(img.astype(np.uint8)) for img in result]
            #image[0].save(f"Depth_Webcam_{time.time()}.png", "PNG")
        #return [depth_image] + [image[0]]
        return image[0]
    def make_batch_sd(
            self,
            image,
            txt,
            device,
            num_samples=1,
            model_type="dpt_hybrid"
    ):
        image = np.array(image)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        #midas_trafo = AddMiDaS(model_type=model_type)
        batch = {
            "jpg": image,
            "txt": num_samples * [txt],
        }
        batch = self.midas_trafo(batch)
        batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
        batch["jpg"] = repeat(batch["jpg"].to(device=device),
                              "1 ... -> n ...", n=num_samples)
        batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
            device=device), "1 ... -> n ...", n=num_samples)
        return batch

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = WebcamWidget()
    widget.show()
    sys.exit(app.exec())
