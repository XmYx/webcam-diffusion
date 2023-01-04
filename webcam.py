import os
import random
import subprocess
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
from matplotlib import pyplot as plt
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
from numpy import float16, uint8
from numpy import array as np_array
#from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS
from PIL.ImageQt import ImageQt
import requests
import cv2
if not os.path.exists("clipseg"):
    cmd = ["git", "clone", "https://github.com/timojl/clipseg"]
    subprocess.Popen(cmd)
sys.path.append("clipseg")
sys.path.append("taming-transformers")
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from io import BytesIO

from torch import autocast
import requests
import PIL
import torch
from diffusers import StableDiffusionInpaintPipeline as StableDiffusionInpaintPipeline
from resizeRight import resizeright, interp_methods
import torchvision.transforms as T
torch.set_grad_enabled(False)
import threading

import cv2
import numpy as np

def blend_images(img1, img2, num_frames=10):
    images = []
    #print(img1, img2)
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
        self.capture_button = QtWidgets.QPushButton('Stop')
        self.capture_button.clicked.connect(self.stop_threads)
        self.continous = QtWidgets.QPushButton('Start')
        self.continous.clicked.connect(self.start_continuous_capture)
        self.webcam_dropdown = QtWidgets.QComboBox()
        self.webcam_dropdown.addItems(self.get_available_webcams())
        self.webcam_dropdown.currentIndexChanged.connect(self.start_webcam)
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(512, 512)

        self.maskprompt = QTextEdit()
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
        self.modelselect = QComboBox()
        self.modelselect.addItems(["Normal", "Depth Model", "Inpaint", "Vanilla Inpaint"])
        # Set up the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.maskprompt)
        layout.addWidget(self.prompt)
        layout.addWidget(self.steps)
        layout.addWidget(self.strength)
        layout.addWidget(self.rescalefactorlabel)
        layout.addWidget(self.rescalefactor)
        layout.addWidget(self.eta)
        layout.addWidget(self.seed)
        layout.addWidget(self.samplercombobox)
        layout.addWidget(self.modelselect)
        layout.addWidget(self.webcam_dropdown)
        layout.addWidget(self.capture_button)
        layout.addWidget(self.continous)
        layout.addWidget(self.camera_label)
        self.setLayout(layout)
        self.threadpool = QThreadPool()
        # Start the webcam
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
        self.loadedmodel = None


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
    def stop_threads(self):
        self.run = False

    def start_continuous_capture(self):

        inference = self.modelselect.currentText()
        if inference == "Normal":
            if self.loadedmodel != "normal":
                self.model = load_model_from_config("configs/stable-diffusion/v1-inference.yaml",
                                                    "models/v1-5-pruned-emaonly.ckpt")
                self.loadedmodel = "normal"
                self.midas_trafo = None
                self.sampler = None
        elif inference == "Depth Model":
            if self.loadedmodel != "depth":
                self.model = initialize_model("configs/stable-diffusion/v2-midas-inference.yaml",
                                           "models/512-depth-ema.ckpt")
                self.sampler = DDIMSampler(self.model)
                model_type = "dpt_hybrid"
                self.midas_trafo = AddMiDaS(model_type=model_type)
                self.loadedmodel = "depth"
        elif inference == "Inpaint":
            if self.loadedmodel != "inpaint":

                self.init_inpaintmasking()
                self.loadedmodel = "inpaint"

                worker2 = Worker(self.continous_mask_thread)
                self.threadpool.start(worker2)
        elif inference == "Vanilla Inpaint":
            if self.loadedmodel != "vanillainpaint":
                self.load_inpaint_model()
                self.init_inpaintmasking()
                self.loadedmodel = "vanillainpaint"

                worker2 = Worker(self.continous_mask_thread)
                self.threadpool.start(worker2)
        """Start the continuous capture in a separate thread."""
        self.run = True
        torch.cuda.empty_cache()
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
    def start_continous_capture_again(self):
        worker2 = Worker(self.continous_mask_thread)
        self.threadpool.start(worker2)

    def make_sampler_schedule(self):
        steps = self.steps.value()
        eta = self.eta.value()
        if self.loadedmodel == 'normal':
            self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    def continous_mask_thread(self, progress_callback=None):
        while self.run == True:
            _, frame = self.capture.read()
            self.frame_to_mask_png(frame)
        if self.run == False:
            return

    def continuous_capture(self, progress_callback=None):
        """Capture images from the webcam continuously."""
        # State for interpolating between diffusion steps
        self.images = []
        self.index = 0
        self.lastinit = None
        self.seedint = 0
        if self.loadedmodel == "inpaint":
            self.model = None
        if self.loadedmodel == "normal":
            with autocast("cuda"):
                self.uc = self.model.get_learned_conditioning(1 * [""])

            self.init_mask_model()
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
            grad_inject_timing_fn = make_inject_timing_fn(1, model_wrap, 10)
            self.cfg_model = CFGDenoiserWithGrad(model_wrap,
                                            loss_fns_scales,
                                            clamp_fn,
                                            None,
                                            "both",
                                            True,
                                            decode_method=None,
                                            grad_inject_timing_fn=grad_inject_timing_fn,
                                            grad_consolidate_fn=None,
                                            verbose=False)
        _, frame = self.capture.read()
        self.frame_to_mask_png(frame)
        while self.run == True:
            #with autocast("cuda"):
            with torch.inference_mode():
                _, frame = self.capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Call the predict function and get the resulting image
                prompt = self.prompt.toPlainText()
                steps = self.steps.value()
                strength = self.strength.value()
                self.seedint = self.seed.text() if self.seed.text() != '' else self.seedint
                self.seedint = int(self.seedint) + 1 if self.seedint != '' else random.randint(0, 4000000)
                eta = self.eta.value()
                if self.loadedmodel == 'depth':
                    self.sampler.make_schedule(steps, ddim_eta=eta, verbose=True)

                if self.loadedmodel == 'normal':
                    result_image = self.img2img(frame, prompt,
                                                steps, 1, 7.5, self.seedint, eta, strength)
                elif self.loadedmodel == 'depth':
                    result_image = self.predict(frame, prompt,
                                                steps, 1, 7.5, self.seedint, eta, strength)
                elif self.loadedmodel == 'inpaint':
                    mask_prompt = self.maskprompt.toPlainText()
                    result_image = self.inpaint_mask_and_replace(frame, mask_prompt, prompt,
                                                steps, 1, 7.5, self.seedint, eta, strength)
                elif self.loadedmodel == 'vanillainpaint':
                    init_image = Image.fromarray(frame)
                    result_image = self.run_vanilla_inpaint(init_image=init_image, prompt=prompt,
                                            seed=self.seedint,
                                            steps=steps,
                                            W=init_image.size[0],
                                            H=init_image.size[1],
                                            outdir='output/outpaint',
                                            n_samples=1,
                                            n_rows=1,
                                            ddim_eta=eta,
                                            blend_mask=None,
                                            mask_blur=0,
                                            recons_blur=0,
                                            strength=strength,
                                            n_iter=1,
                                            scale=7.5,
                                            skip_save=True,
                                            skip_grid=True,
                                            file_prefix="outpaint",
                                            image_callback=None,
                                            step_callback=None,
                                            with_inpaint=True)

                # Store the result image in a list
                #result_images.append(result_image[1])
                self.images.append(result_image)
                self.morphed_images = []
                if len(self.images) > 1:
                    self.index = 0
                    self.morphed_images = blend_images(self.images[len(self.images) - 2], self.images[len(self.images) - 1])
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
        if self.run == True:
            self.index += 1
            donotdraw = None
            if self.index >= len(self.morphed_images) - 1:
                self.index = len(self.morphed_images) - 1
                donotdraw = True
            if self.morphed_images != []:
                if donotdraw is not True:
                    self.image_label.setPixmap(QtGui.QPixmap.fromImage(ImageQt(Image.fromarray(self.morphed_images[self.index]))))
                    self.image_label.setScaledContents(True)
        else:
            self.index = 0
            return
    def init_inpaintmasking(self):
        self.init_mask_model()
        device = "cuda"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=""
        ).to(device)
        self.pipe.enable_xformers_memory_efficient_attention()
        #torch.backends.cudnn.benchmark = True
        self.pipe.safety_checker = dummy
        torch.backends.cuda.matmul.allow_tf32 = True
    def init_mask_model(self):
        self.maskmodel = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
        self.maskmodel.eval()
        self.maskmodel.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict=False)
    def frame_to_mask_png(self, frame):
        #self.mask = Image.open('mask.png')
        input_image = Image.fromarray(frame)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((512, 512)),
        ])
        img = transform(input_image).unsqueeze(0)
        #input_image.convert("RGB").resize((512, 512)).save("init_image.png", "PNG")
        prompts = [self.maskprompt.toPlainText()]
        # predict
        with torch.no_grad():
            preds = self.maskmodel(img.repeat(len(prompts), 1, 1, 1), prompts)[0]
        # Convert the image data to a NumPy array
        image_data = preds[0][0].numpy()
        # Normalize the image data between 0 and 255
        #image_data = (image_data * 255).astype(np.uint8)
        transform = transforms.ToPILImage()
        self.mask = transform(torch.sigmoid(preds[0][0]))
        # Convert the image data to a PIL.Image object
        #self.mask = Image.fromarray(preds)
        self.mask.save("mask.png")
        #filename = f"mask.png"
        #plt.imsave(filename, torch.sigmoid(preds[0][0]))

        #self.mask = Image.open('mask.png')
    def inpaint_mask_and_replace(self, frame, mask_prompt, prompt, steps, n_samples, scale, seed, eta, strength):

        input_image = Image.fromarray(frame)
        init_image = input_image.convert("RGB").resize((512, 512))
        mask = self.mask.resize((512,512))
        images = self.pipe(prompt=self.prompt.toPlainText(), num_inference_steps=self.steps.value(), image=init_image, mask_image=mask)['images']
        return images[0]
    def img2img(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        image = input_image.astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.*image - 1.
        image = image.half().to("cuda")
        data = [1 * [prompt]]
        t_enc = int(strength * steps)
        seed_everything(seed)
        factor = self.rescalefactor.value()
        with torch.no_grad():
            #torch.backends.cudnn.benchmark = True
            with autocast("cuda"):
                with self.model.ema_scope():
                    init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(image))
                    if factor != 1.0:
                        init_latent = resizeright.resize(init_latent, scale_factors=None,
                                                     out_shape=[init_latent.shape[0], init_latent.shape[1], int(init_latent.shape[2] // factor),
                                                                int(init_latent.shape[3] // factor)],
                                                     interp_method=interp_methods.lanczos2, support_sz=None,
                                                     antialiasing=False, by_convs=True, scale_tolerance=None,
                                                     max_numerator=10, pad_mode='reflect')

                    #tic = time.time()
                    #all_samples = list()
                    #for n in trange(1, desc="Sampling"):
                    #    for prompts in tqdm(data, desc="data"):
                    prompts = prompt
                    if isinstance(prompt, tuple):
                        prompts = list(prompt)
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
                    c = self.model.get_learned_conditioning(prompts)
                    samples = sampler_fn(
                        c=c,
                        uc=self.uc,
                        args=args,
                        model_wrap=self.cfg_model,
                        init_latent=init_latent,
                        t_enc=t_enc,
                        device="cuda",
                        cb=None,
                        verbose=False)


                    x_samples = self.model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                    return image

    def predict(self, input_image, prompt, steps, num_samples, scale, seed, eta, strength):
        do_full_sample = strength == 1.
        t_enc = min(int(strength * steps), steps-1)
        input_image = Image.fromarray(input_image)
        width, height = input_image.size
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
        return result


    def paint(self, sampler, model, image, image_quad, prompt, t_enc, seed, scale, num_samples=1, callback=None,
              do_full_sample=False):
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        seed_everything(seed)
        with torch.no_grad(),\
                torch.autocast("cuda"):
            batch = self.make_batch_sd(
                image, txt=prompt, device=device, num_samples=num_samples)
            z = model.get_first_stage_encoding(model.encode_first_stage(
                batch[model.first_stage_key]))
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck]
                cc = model.depth_model(cc)
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
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            z_enc = torch.randn_like(z)
            # decode it

            samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc_full, callback=callback)
            x_samples_ddim = model.decode_first_stage(samples)
            result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
            image = [Image.fromarray(img.astype(np.uint8)) for img in result]
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
    def load_inpaint_model(self):

            weights = 'models/sd-v1-5-inpainting.ckpt'
            config = 'configs/stable-diffusion/inpaint.yaml'
            embedding_path = None

            config = OmegaConf.load(config)

            self.model = instantiate_from_config(config.model)

            self.model.load_state_dict(torch.load(weights)["state_dict"], strict=False)

            device = "cuda"
            self.model.half().to(device)
            self.loadedmodel = 'vanillainpaint'
            return
    def run_vanilla_inpaint(self,
                         init_image,
                         prompt="fantasy landscape",
                         seed=-1,
                         steps=10,
                         W=512,
                         H=512,
                         outdir='output/outpaint',
                         n_samples=1,
                         n_rows=1,
                         ddim_eta=0.0,
                         blend_mask=None,
                         mask_blur=0,
                         recons_blur=0,
                         strength=0.95,
                         n_iter=1,
                         scale=7,
                         skip_save=False,
                         skip_grid=True,
                         file_prefix="outpaint",
                         image_callback=None,
                         step_callback=None,
                         with_inpaint=True,
                         ):
        print("Using 1.5 InPaint model") if with_inpaint else None
        mask_img = self.mask.resize(init_image.size, resample=Image.Resampling.NEAREST)
        img = init_image
        # mask_img = img.split()[-1]
        # print(f"using seed: {seed}")
        if seed == 0 or seed == -1 or seed == '':
            seed = seed_everything()
        width = img.size[0]
        height = img.size[1]
        if mask_blur > 0 and with_inpaint == True:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))

        os.makedirs('output/temp', exist_ok=True)
        mask_img.save('output/temp/mask.png')
        blend_mask = 'output/temp/mask.png'
        os.makedirs(outdir, exist_ok=True)
        outpath = outdir
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        base_name = f"{random.randint(10000000, 99999999)}_{seed}_"

        print(F"WITH INPAINT : {with_inpaint}")

        if self.loadedmodel != 'vanillainpaint':
            self.load_inpaint_model()
        sampler = DDIMSampler(self.model)
        source_w, source_h = init_image.size

        #w, h = map(lambda x: x - x % 64, (source_w, source_h))  # resize to integer multiple of 32
        #if source_w != w or source_h != h:
        #    image = init_image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np_array(init_image).astype(float16) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).half().to("cuda")

        #image_guide = image
        [mask_for_reconstruction, latent_mask_for_blend] = get_mask_for_latent_blending("cuda", blend_mask,
                                                                                        blur=mask_blur,
                                                                                        recons_blur=recons_blur)
        masked_image_for_blend = (1 - mask_for_reconstruction) * image[0]

        mask = mask_img
        #image = Image.open(init_image)
        result = vanilla_inpaint(
            model=self.model,
            sampler=sampler,
            image=init_image,
            mask=mask,
            prompt=prompt,
            seed=seed,
            scale=scale,
            ddim_steps=steps,
            num_samples=1,
            h=height, w=width,
            device="cuda",
            mask_for_reconstruction=mask_for_reconstruction,
            masked_image_for_blend=masked_image_for_blend,
            callback=step_callback)
        fpath = os.path.join(sample_path, f"{base_name}_{base_count:05}.png")
        result[0].save(fpath, 'PNG')

        return result[0]




def dummy(images, **kwargs):
    return images, False



def vanilla_inpaint(model, sampler, image, mask, prompt, seed, scale, ddim_steps, device, mask_for_reconstruction,
            masked_image_for_blend, num_samples=1, w=512, h=512, callback=None):
    # model = sampler.model
    # model.to(device)

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float16)

    # model.model.to("cpu")
    # model.cond_stage_model.to(device)
    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h // 8, w // 8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            # model.cond_stage_model.to("cpu")
            # model.model.to(device)
            shape = [model.channels, h // 8, w // 8]
            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
                img_callback=callback,
            )
            x_samples = encoded_to_torch_image(
                model, samples_cfg)  # [1, 3, 512, 512]
            all_samples = []
            if masked_image_for_blend is not None:
                x_samples = mask_for_reconstruction * x_samples + masked_image_for_blend

            all_samples.append(x_samples)

            generated_time = time.time()

            for x_sample in x_samples:
                image = sampleToImage(x_sample)
                result = [image]

                # image.save(os.path.join(sample_path, f"{base_count:05}.png"))
                # if image_callback is not None:
                #    image_callback(image)

            # result = torch.clamp((x_samples+1.0)/2.0,
            #                     min=0.0, max=1.0)

            # result = result.cpu().numpy().transpose(0,2,3,1)
            # result = result*255

    # result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    # result = [put_watermark(img for img in result]
    # model.to("cpu")
    return result


def torch_image_to_latent(model, torch_image, n_samples=1):
    formatted_image = 2. * torch_image - 1.
    if n_samples > 1:
        formatted_image = repeat(
            formatted_image, '1 ... -> b ...', b=n_samples)
    latent_image = model.get_first_stage_encoding(
        model.encode_first_stage(formatted_image))
    return latent_image


# [1, 4, 64, 64] => [1, 3, 512, 512]
def encoded_to_torch_image(model, encoded_image):
    decoded = model.decode_first_stage(encoded_image)
    return torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)


# [1, 4, 64, 64] => image
def encoded_to_image(model, encoded_image):
    return sampleToImage(encoded_to_torch_image(model, encoded_image)[0])


def sampleToImage (sample):
    sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(sample.astype(uint8))

def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

def get_mask_for_latent_blending(device, path, blur = 0, recons_blur=0):
    mask_image = Image.open(path).convert("L")

    if blur > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(blur))

    mask_for_reconstruction = mask_image.point(lambda x: 255 if x > 0 else 0)
    if recons_blur > 0:
        mask_for_reconstruction = mask_for_reconstruction.filter(
            ImageFilter.GaussianBlur(radius=recons_blur))
    mask_for_reconstruction = mask_for_reconstruction.point(
        lambda x: 255 if x > 127 else x * 2)

    mask_for_reconstruction = torch.from_numpy(
        (np_array(mask_for_reconstruction) / 255.0).astype(float16)).to(device)

    source_w, source_h = mask_image.size


    mask = np_array(
        mask_image.resize(
            (int(source_w / 8), int(source_h / 8)), resample=Image.Resampling.LANCZOS).convert("L"))
    mask = (mask / 255.0).astype(float16)

    mask = mask[None]
    mask = 1 - mask

    mask = torch.from_numpy(mask)

    mask = torch.stack([mask, mask, mask, mask], 1).to(device)  # FIXME
    return [mask_for_reconstruction, mask]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    widget = WebcamWidget()
    widget.show()
    sys.exit(app.exec())
