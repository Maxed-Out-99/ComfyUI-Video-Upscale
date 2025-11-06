# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import contextlib
import json
import logging
import math
from enum import Enum
from importlib import import_module
from pathlib import Path

import torch
import comfy
from PIL import Image, ImageFilter
from comfy_extras.nodes_custom_sampler import SamplerCustom
from tqdm import tqdm

from usdu_patch import usdu
from utils import tensor_to_pil, pil_to_tensor, get_crop_region, expand_crop, crop_cond
import modules.shared as shared
from modules.upscaler import UpscalerData

comfy_nodes = import_module("nodes")
common_ksampler = comfy_nodes.common_ksampler
VAEEncode = comfy_nodes.VAEEncode
VAEDecode = comfy_nodes.VAEDecode
VAEDecodeTiled = comfy_nodes.VAEDecodeTiled

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = None
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            print("[USDU] Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            print("[USDU] Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = Path(__file__).with_name('config.json')
        config = {}
        if config_path.exists():
            with config_path.open('r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if comfy.utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy.utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar = None
            # self.pbar = tqdm(total=self.tiles, desc='USDU') # Creating the pbar here will cause an empty progress bar to be displayed

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy.utils.PROGRESS_BAR_ENABLED = True


class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    # Choose way to sample based on given inputs

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        custom_sample = SamplerCustom()
        (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        return samples

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Crop conditioning
    positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
    negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # Generate samples
    samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                     negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed
@contextlib.contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)

INPUT_TYPES = {
    "required": {
        "image": ("IMAGE",),
        # Sampling Params
        "model": ("MODEL",),
        "positive": ("CONDITIONING",),
        "negative": ("CONDITIONING",),
        "vae": ("VAE",),
        "upscale_by": ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
        "denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
        # Upscale Params
        "upscale_model": ("UPSCALE_MODEL",),
        "mode_type": (list(MODES.keys()),),
        "tile_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
        "tile_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
        "mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
        "tile_padding": ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        # Seam fix params
        "seam_fix_mode": (list(SEAM_FIX_MODES.keys()),),
        "seam_fix_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        "seam_fix_mask_blur": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
        "seam_fix_padding": ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
        # Misc
        "force_uniform_tiles": ("BOOLEAN", {"default": True}),
        "tiled_decode": ("BOOLEAN", {"default": False}),
    }
}


class VideoUpscalerMXD:
    @classmethod
    def INPUT_TYPES(s):
        return INPUT_TYPES

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, 
                custom_sampler=None, custom_sigmas=None):
        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, MODES[self.mode_type], SEAM_FIX_MODES[self.seam_fix_mode],
            custom_sampler, custom_sigmas,
        )

        # Disable logging
        with suppress_logging():
            # Running the script
            script = usdu.Script()
            script.run(
                p=sdprocessing,
                _=None,
                tile_width=tile_width,
                tile_height=tile_height,
                mask_blur=mask_blur,
                padding=tile_padding,
                seams_fix_width=seam_fix_width,
                seams_fix_denoise=seam_fix_denoise,
                seams_fix_padding=seam_fix_padding,
                upscaler_index=0,
                save_upscaled_image=False,
                redraw_mode=MODES[mode_type],
                save_seams_fix_image=False,
                seams_fix_mask_blur=seam_fix_mask_blur,
                seams_fix_type=SEAM_FIX_MODES[seam_fix_mode],
                target_size_type=2,
                custom_width=None,
                custom_height=None,
                custom_scale=upscale_by,
            )

            # Return the resulting images
            images = [pil_to_tensor(img) for img in shared.batch]
            tensor = torch.cat(images, dim=0)
            return (tensor,)

NODE_CLASS_MAPPINGS = {
    "VideoUpscalerMXD": VideoUpscalerMXD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoUpscalerMXD": "Video Upscaler MXD",
}
