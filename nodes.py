# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import contextlib
import logging
import math
from enum import Enum

import torch
import comfy
from PIL import Image, ImageDraw, ImageOps

from utils import tensor_to_pil, pil_to_tensor
from modules.processing import Processed, StableDiffusionProcessing, fix_seed, process_images
import modules.shared as shared
from modules.upscaler import UpscalerData

if not hasattr(Image, "Resampling"):
    Image.Resampling = Image


def flatten(img, bgcolor):
    if img.mode in ("RGB"):
        return img
    return Image.alpha_composite(Image.new("RGBA", img.size, bgcolor), img).convert("RGB")


def torch_gc():
    pass


class ScriptBase:
    pass


def round_length(length, multiple=8):
    return round(length / multiple) * multiple


class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2


class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class USDURedraw:
    def __init__(self):
        self.initial_info = None
        self.mode = USDUMode.NONE

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        processed = None
        for yi in range(rows):
            for xi in range(cols):
                if shared.state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if processed.images:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def chess_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        processed = None
        tiles = []
        for yi in range(rows):
            row = []
            for xi in range(cols):
                if shared.state.interrupted:
                    break
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                row.append(color)
            tiles.append(row)

        for yi, row in enumerate(tiles):
            for xi, enabled in enumerate(row):
                if shared.state.interrupted:
                    break
                if not enabled:
                    continue
                tiles[yi][xi] = False
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if processed.images:
                    image = processed.images[0]

        for yi, row in enumerate(tiles):
            for xi, enabled in enumerate(row):
                if shared.state.interrupted:
                    break
                if enabled:
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if processed.images:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)
        return image


class USDUSeamsFix:
    def __init__(self):
        self.initial_info = None
        self.mode = USDUSFMode.NONE

    def init_draw(self, p):
        self.initial_info = None
        p.width = round_length(self.tile_width + self.padding)
        p.height = round_length(self.tile_height + self.padding)

    def half_tile_process(self, p, image, rows, cols):
        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(
            gradient.resize((self.tile_width, self.tile_height // 2), resample=Image.BICUBIC),
            (0, 0),
        )
        row_gradient.paste(
            gradient.rotate(180).resize((self.tile_width, self.tile_height // 2), resample=Image.BICUBIC),
            (0, self.tile_height // 2),
        )
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(
            gradient.rotate(90).resize((self.tile_width // 2, self.tile_height), resample=Image.BICUBIC),
            (0, 0),
        )
        col_gradient.paste(
            gradient.rotate(270).resize((self.tile_width // 2, self.tile_height), resample=Image.BICUBIC),
            (self.tile_width // 2, 0),
        )

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols):
                if shared.state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi * self.tile_width, yi * self.tile_height + self.tile_height // 2))

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if processed.images:
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols - 1):
                if shared.state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi * self.tile_width + self.tile_width // 2, yi * self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = process_images(p)
                if processed.images:
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize((self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows - 1):
            for xi in range(cols - 1):
                if shared.state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(
                    gradient,
                    (xi * self.tile_width + self.tile_width // 2, yi * self.tile_height + self.tile_height // 2),
                )

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = process_images(p)
                if processed.images:
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, rows, cols):
        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if shared.state.interrupted:
                break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if processed.images:
                image = processed.images[0]

        for yi in range(1, cols):
            if shared.state.interrupted:
                break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = process_images(p)
            if processed.images:
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        mode = self.mode
        if mode == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        if mode == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        if mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        return image


class USDUpscaler:
    def __init__(self, p, image, upscaler_index, save_redraw, save_seams_fix, tile_width, tile_height):
        p.width = round_length(image.width * p.upscale_by)
        p.height = round_length(image.height * p.upscale_by)

        self.p = p
        self.image = image
        self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None
        self.rows = math.ceil(self.p.height / self.redraw.tile_height)
        self.cols = math.ceil(self.p.width / self.redraw.tile_width)

    def get_factor(self, num):
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        print(f"Canva size: {self.p.width}x{self.p.height}")
        print(f"Image size: {self.image.width}x{self.image.height}")
        print(f"Scale factor: {self.scale_factor}")
        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.Resampling.LANCZOS)
        else:
            self.get_factors()
            for index, value in self.scales:
                print(f"Upscaling iteration {index + 1} with scale factor {value}")
                self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.Resampling.LANCZOS)

        additional = []
        if shared.batch is not None and len(shared.batch) > 1:
            additional = [
                img.resize((self.p.width, self.p.height), resample=Image.Resampling.LANCZOS)
                for img in shared.batch[1:]
            ]
        shared.batch = [self.image] + additional

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        self.redraw.mode = redraw_mode
        self.redraw.enabled = self.redraw.mode != USDUMode.NONE
        self.redraw.padding = padding
        self.p.mask_blur = mask_blur

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = mode
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        # Saving is handled by Automatic1111 in the original script.
        # The lightweight ComfyUI integration does not persist images to disk.
        pass

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if getattr(self.redraw, "enabled", False) else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        shared.state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {getattr(self.redraw, 'enabled', False)}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        shared.state.begin()
        self.calc_jobs_count()
        self.result_images = []
        if getattr(self.redraw, "enabled", False):
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if getattr(self.redraw, "save", False):
            self.save_image()

        if getattr(self.seams_fix, "enabled", False):
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if getattr(self.seams_fix, "save", False):
                self.save_image()
        shared.state.end()

        if shared.batch:
            shared.batch[0] = self.image


class Script(ScriptBase):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def run(
        self,
        p,
        _,
        tile_width,
        tile_height,
        mask_blur,
        padding,
        seams_fix_width,
        seams_fix_denoise,
        seams_fix_padding,
        upscaler_index,
        save_upscaled_image,
        redraw_mode,
        save_seams_fix_image,
        seams_fix_mask_blur,
        seams_fix_type,
        target_size_type,
        custom_width,
        custom_height,
        custom_scale,
    ):
        fix_seed(p)
        torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        init_img = p.init_images[0]
        if init_img is None:
            return Processed(p, [], seed, "Empty image")
        init_img = flatten(init_img, shared.opts.img2img_background_color)

        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = round_length(init_img.width * custom_scale)
            p.height = round_length(init_img.height * custom_scale)

        upscaler = USDUpscaler(
            p,
            init_img,
            upscaler_index,
            save_upscaled_image,
            save_seams_fix_image,
            tile_width,
            tile_height,
        )
        upscaler.upscale()

        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type)
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}
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
            shared.batch[0],
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
            force_uniform_tiles,
            tiled_decode,
            tile_width,
            tile_height,
            MODES[mode_type],
            SEAM_FIX_MODES[seam_fix_mode],
            custom_sampler,
            custom_sigmas,
        )

        # Disable logging
        with suppress_logging():
            # Running the script
            script = Script()
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
