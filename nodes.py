import contextlib
import json
import logging
import math
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps
from types import SimpleNamespace
from enum import Enum
from importlib import import_module
from pathlib import Path
import torch
import torch.nn.functional as F
import comfy
from comfy_extras.nodes_custom_sampler import SamplerCustom
from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
from tqdm import tqdm
from usdu_patch import usdu
from utils import tensor_to_pil, pil_to_tensor, get_crop_region, expand_crop, crop_cond


# Compatibility fix for older Pillow versions
if not hasattr(Image, "Resampling"):
    Image.Resampling = Image


# Compatibility fix for older Pillow versions
if not hasattr(Image, "Resampling"):
    Image.Resampling = Image


# Mimic A1111 shared objects (needed by some Ultimate SD Upscale functions)
class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


shared = SimpleNamespace(
    Options=Options,
    State=State,
    opts=Options(),
    state=State(),
    sd_upscalers=[None],
    actual_upscaler=None,
    batch=None,
    batch_as_tensor=None,
)

comfy_nodes = import_module("nodes")
common_ksampler = comfy_nodes.common_ksampler
VAEEncode = comfy_nodes.VAEEncode
VAEDecode = comfy_nodes.VAEDecode
VAEDecodeTiled = comfy_nodes.VAEDecodeTiled

if not hasattr(Image, "Resampling"):  # For older versions of Pillow
    Image.Resampling = Image

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



def tensor_to_pil(img_tensor, batch_index=0):
    # Takes a batch of images in the form of a tensor of shape [batch_size, height, width, channels]
    # and returns an RGB PIL Image. Assumes channels=3
    safe_tensor = torch.nan_to_num(img_tensor[batch_index])
    return Image.fromarray((255 * safe_tensor.cpu().numpy()).astype(np.uint8))


def pil_to_tensor(image):
    # Takes a PIL image and returns a tensor of shape [1, height, width, channels]
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0)
    if len(image.shape) == 3:  # If the image is grayscale, add a channel dimension
        image = image.unsqueeze(-1)
    return image


def crop_tensor(tensor, region):
    # Takes a tensor of shape [batch_size, height, width, channels] and crops it to the given region
    x1, y1, x2, y2 = region
    return tensor[:, y1:y2, x1:x2, :]


def resize_tensor(tensor, size, mode="nearest-exact"):
    # Takes a tensor of shape [B, C, H, W] and resizes
    # it to a shape of [B, C, size[0], size[1]] using the given mode
    return torch.nn.functional.interpolate(tensor, size=size, mode=mode)


def get_crop_region(mask, pad=0):
    # Takes a black and white PIL image in 'L' mode and returns the coordinates of the white rectangular mask region
    # Should be equivalent to the get_crop_region function from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/masking.py
    coordinates = mask.getbbox()
    if coordinates is not None:
        x1, y1, x2, y2 = coordinates
    else:
        x1, y1, x2, y2 = mask.width, mask.height, 0, 0
    # Apply padding
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    return fix_crop_region((x1, y1, x2, y2), (mask.width, mask.height))


def fix_crop_region(region, image_size):
    # Remove the extra pixel added by the get_crop_region function
    image_width, image_height = image_size
    x1, y1, x2, y2 = region
    if x2 < image_width:
        x2 -= 1
    if y2 < image_height:
        y2 -= 1
    return x1, y1, x2, y2


def expand_crop(region, width, height, target_width, target_height):
    """
    Expands a crop region to a specified target size.
    :param region: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param width: The width of the image the crop region is from.
    :param height: The height of the image the crop region is from.
    :param target_width: The desired width of the crop region.
    :param target_height: The desired height of the crop region.
    """
    x1, y1, x2, y2 = region
    actual_width = x2 - x1
    actual_height = y2 - y1

    # Try to expand region to the right of half the difference
    width_diff = target_width - actual_width
    x2 = min(x2 + width_diff // 2, width)
    # Expand region to the left of the difference including the pixels that could not be expanded to the right
    width_diff = target_width - (x2 - x1)
    x1 = max(x1 - width_diff, 0)
    # Try the right again
    width_diff = target_width - (x2 - x1)
    x2 = min(x2 + width_diff, width)

    # Try to expand region to the bottom of half the difference
    height_diff = target_height - actual_height
    y2 = min(y2 + height_diff // 2, height)
    # Expand region to the top of the difference including the pixels that could not be expanded to the bottom
    height_diff = target_height - (y2 - y1)
    y1 = max(y1 - height_diff, 0)
    # Try the bottom again
    height_diff = target_height - (y2 - y1)
    y2 = min(y2 + height_diff, height)

    return (x1, y1, x2, y2), (target_width, target_height)


def resize_region(region, init_size, resize_size):
    # Resize a crop so that it fits an image that was resized to the given width and height
    x1, y1, x2, y2 = region
    init_width, init_height = init_size
    resize_width, resize_height = resize_size
    x1 = math.floor(x1 * resize_width / init_width)
    x2 = math.ceil(x2 * resize_width / init_width)
    y1 = math.floor(y1 * resize_height / init_height)
    y2 = math.ceil(y2 * resize_height / init_height)
    return x1, y1, x2, y2


def pad_image2(image, left_pad, right_pad, top_pad, bottom_pad, fill=False, blur=False):
    """
    Pads an image with the given number of pixels on each side and fills the padding with data from the edges.
    Faster than pad_image, but only pads with edge data in straight lines.
    :param image: A PIL image
    :param left_pad: The number of pixels to pad on the left side
    :param right_pad: The number of pixels to pad on the right side
    :param top_pad: The number of pixels to pad on the top side
    :param bottom_pad: The number of pixels to pad on the bottom side
    :param blur: Whether to blur the padded edges
    :return: A PIL image with size (image.width + left_pad + right_pad, image.height + top_pad + bottom_pad)
    """
    left_edge = image.crop((0, 1, 1, image.height - 1))
    right_edge = image.crop((image.width - 1, 1, image.width, image.height - 1))
    top_edge = image.crop((1, 0, image.width - 1, 1))
    bottom_edge = image.crop((1, image.height - 1, image.width - 1, image.height))
    new_width = image.width + left_pad + right_pad
    new_height = image.height + top_pad + bottom_pad
    padded_image = Image.new(image.mode, (new_width, new_height))
    padded_image.paste(image, (left_pad, top_pad))
    if fill:
        if left_pad > 0:
            padded_image.paste(left_edge.resize((left_pad, new_height), resample=Image.Resampling.NEAREST), (0, 0))
        if right_pad > 0:
            padded_image.paste(
                right_edge.resize((right_pad, new_height), resample=Image.Resampling.NEAREST),
                (new_width - right_pad, 0),
            )
        if top_pad > 0:
            padded_image.paste(
                top_edge.resize((new_width, top_pad), resample=Image.Resampling.NEAREST),
                (0, 0),
            )
        if bottom_pad > 0:
            padded_image.paste(
                bottom_edge.resize((new_width, bottom_pad), resample=Image.Resampling.NEAREST),
                (0, new_height - bottom_pad),
            )
    return padded_image


def resize_and_pad_image(image, width, height, fill=False, blur=False):
    """
    Resizes an image to the closest size that maintains the aspect ratio and pads it to the given width and height.
    :param image: A PIL image
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param fill: Whether to fill the padding with data from the edges
    :param blur: Whether to blur the padded edges
    :return: A PIL image of size (width, height)
    """
    width_ratio = width / image.width
    height_ratio = height / image.height
    if height_ratio > width_ratio:
        resize_ratio = width_ratio
    else:
        resize_ratio = height_ratio
    resize_width = round(image.width * resize_ratio)
    resize_height = round(image.height * resize_ratio)
    resized = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)
    # Pad the sides of the image to get the image to the desired size that wasn't covered by the resize
    horizontal_pad = (width - resize_width) // 2
    vertical_pad = (height - resize_height) // 2
    result = pad_image2(resized, horizontal_pad, horizontal_pad, vertical_pad, vertical_pad, fill, blur)
    result = result.resize((width, height), resample=Image.Resampling.LANCZOS)
    return result, (horizontal_pad, vertical_pad)


def region_intersection(region1, region2):
    """
    Returns the coordinates of the intersection of two rectangular regions.
    :param region1: A tuple of the form (x1, y1, x2, y2) denoting the upper left and the lower right points
        of the first rectangular region. Expected to have x2 > x1 and y2 > y1.
    :param region2: The second rectangular region with the same format as the first.
    :return: A tuple of the form (x1, y1, x2, y2) denoting the rectangular intersection.
        None if there is no intersection.
    """
    x1, y1, x2, y2 = region1
    x1_, y1_, x2_, y2_ = region2
    x1 = max(x1, x1_)
    y1 = max(y1, y1_)
    x2 = min(x2, x2_)
    y2 = min(y2, y2_)
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "control" not in cond_dict:
        return
    c = cond_dict["control"]
    controlnet = c.copy()
    cond_dict["control"] = controlnet
    while c is not None:
        # hint is shape (B, C, H, W)
        hint = controlnet.cond_hint_original
        resized_crop = resize_region(region, canvas_size, hint.shape[:-3:-1])
        hint = crop_tensor(hint.movedim(1, -1), resized_crop).movedim(-1, 1)
        hint = resize_tensor(hint, tile_size[::-1])
        controlnet.cond_hint_original = hint
        c = c.previous_controlnet
        controlnet.set_previous_controlnet(c.copy() if c is not None else None)
        controlnet = controlnet.previous_controlnet


def crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "gligen" not in cond_dict:
        return
    type, model, cond = cond_dict["gligen"]
    if type != "position":
        from warnings import warn

        warn(f"Unknown gligen type {type}")
        return
    cropped = []
    for c in cond:
        emb, h, w, y, x = c
        # Get the coordinates of the box in the upscaled image
        x1 = x * 8
        y1 = y * 8
        x2 = x1 + w * 8
        y2 = y1 + h * 8
        gligen_upscaled_box = resize_region((x1, y1, x2, y2), init_size, canvas_size)

        # Calculate the intersection of the gligen box and the region
        intersection = region_intersection(gligen_upscaled_box, region)
        if intersection is None:
            continue
        x1, y1, x2, y2 = intersection

        # Offset the gligen box so that the origin is at the top left of the tile region
        x1 -= region[0]
        y1 -= region[1]
        x2 -= region[0]
        y2 -= region[1]

        # Add the padding
        x1 += w_pad
        y1 += h_pad
        x2 += w_pad
        y2 += h_pad

        # Set the new position params
        h = (y2 - y1) // 8
        w = (x2 - x1) // 8
        x = x1 // 8
        y = y1 // 8
        cropped.append((emb, h, w, y, x))

    cond_dict["gligen"] = (type, model, cropped)


def crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "area" not in cond_dict:
        return

    # Resize the area conditioning to the canvas size and confine it to the tile region
    h, w, y, x = cond_dict["area"]
    w, h, x, y = 8 * w, 8 * h, 8 * x, 8 * y
    x1, y1, x2, y2 = resize_region((x, y, x + w, y + h), init_size, canvas_size)
    intersection = region_intersection((x1, y1, x2, y2), region)
    if intersection is None:
        del cond_dict["area"]
        del cond_dict["strength"]
        return
    x1, y1, x2, y2 = intersection

    # Offset origin to the top left of the tile
    x1 -= region[0]
    y1 -= region[1]
    x2 -= region[0]
    y2 -= region[1]

    # Add the padding
    x1 += w_pad
    y1 += h_pad
    x2 += w_pad
    y2 += h_pad

    # Set the params for tile
    w, h = (x2 - x1) // 8, (y2 - y1) // 8
    x, y = x1 // 8, y1 // 8

    cond_dict["area"] = (h, w, y, x)


def crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    if "mask" not in cond_dict:
        return
    mask_tensor = cond_dict["mask"]  # (B, H, W)
    masks = []
    for i in range(mask_tensor.shape[0]):
        # Convert to PIL image
        mask = tensor_to_pil(mask_tensor, i)  # W x H

        # Resize the mask to the canvas size
        mask = mask.resize(canvas_size, Image.Resampling.BICUBIC)

        # Crop the mask to the region
        mask = mask.crop(region)

        # Add padding
        mask, _ = resize_and_pad_image(mask, tile_size[0], tile_size[1], fill=True)

        # Resize the mask to the tile size
        if tile_size != mask.size:
            mask = mask.resize(tile_size, Image.Resampling.BICUBIC)

        # Convert back to tensor
        mask = pil_to_tensor(mask)  # (1, H, W, 1)
        mask = mask.squeeze(-1)  # (1, H, W)
        masks.append(mask)

    cond_dict["mask"] = torch.cat(masks, dim=0)  # (B, H, W)


def crop_reference_latents(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad):
    """
    1. Resize each latent to `canvas_size` in latent units.
    2. Crop the rectangle `region` (pixel coordinates).
    3. Down-sample the crop to latent-space `tile_size`.
    Expects a list of BCHW tensors under "reference_latents".
    """

    latents = cond_dict.get("reference_latents")
    if not isinstance(latents, list):
        return  # nothing to do

    k = 8  # down-sample factor from pixel space → latent space (SD-type models)

    W_can_px, H_can_px = canvas_size
    # canvas size expressed in latent units
    W_can_lat, H_can_lat = W_can_px // k, H_can_px // k

    W_tile_px, H_tile_px = tile_size
    W_tile_lat, H_tile_lat = max(1, W_tile_px // k), max(1, H_tile_px // k)

    x1_px, y1_px, x2_px, y2_px = region

    new_latents = []
    for t in latents:  # (B,C,H_lat_in,W_lat_in)
        has_5d = False
        if t.ndim == 5:  # (B,C,1,H_lat_in,W_lat_in)
            has_5d = True
            t = t.squeeze(2)
        if t.ndim != 4:
            raise ValueError(f"expected BCHW, got {t.shape}")

        # 1. Resize to canvas resolution in latent units only if needed
        if t.shape[-2:] != (H_can_lat, W_can_lat):
            t = F.interpolate(
                t,
                size=(H_can_lat, W_can_lat),
                mode="bilinear",
                align_corners=False,
            )

        # 2. Convert pixel crop → latent slice
        w0_lat = int(round(x1_px / k))
        w1_lat = int(round(x2_px / k))
        h0_lat = int(round(y1_px / k))
        h1_lat = int(round(y2_px / k))

        cropped = t[:, :, h0_lat:h1_lat, w0_lat:w1_lat]  # view

        # 3. Down-sample to latent-tile size
        cropped = F.interpolate(
            cropped,
            size=(H_tile_lat, W_tile_lat),
            mode="bilinear",
            align_corners=False,
        )
        if has_5d:
            cropped = cropped.unsqueeze(2)
        new_latents.append(cropped)

    cond_dict["reference_latents"] = new_latents


def crop_cond(cond, region, init_size, canvas_size, tile_size, w_pad=0, h_pad=0):
    cropped = []
    for emb, x in cond:
        cond_dict = x.copy()
        n = [emb, cond_dict]
        crop_controlnet(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_gligen(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_area(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_mask(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        crop_reference_latents(cond_dict, region, init_size, canvas_size, tile_size, w_pad, h_pad)
        cropped.append(n)
    return cropped
MODES = {
    "Linear": USDUMode.LINEAR,
    "Chess": USDUMode.CHESS,
    "None": USDUMode.NONE,
}

SEAM_FIX_MODES = {
    "None": USDUSFMode.NONE,
    "Band Pass": USDUSFMode.BAND_PASS,
    "Half Tile": USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


# --- A1111 COMPATIBILITY WRAPPERS ------------------------------------------

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

        # Internal helpers
        self.init_size = (init_img.width, init_img.height)
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()
        self.extra_generation_params = {}

        if self.tiled_decode:
            print("[USDU] Using tiled decode")

        # Load optional config
        config_path = Path(__file__).with_name("config.json")
        self.progress_bar_enabled = False
        if config_path.exists():
            with config_path.open("r") as f:
                cfg_json = json.load(f)
            if comfy.utils.PROGRESS_BAR_ENABLED:
                self.progress_bar_enabled = True
                comfy.utils.PROGRESS_BAR_ENABLED = cfg_json.get("per_tile_progress", True)

    def __del__(self):
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


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
           latent, denoise, custom_sampler, custom_sigmas):
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
            latent_image=latent,
        )
        return samples

    (samples,) = common_ksampler(
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=denoise,
    )
    return samples


def process_images(p: StableDiffusionProcessing) -> Processed:
    if p.progress_bar_enabled and getattr(p, "pbar", None) is None:
        p.pbar = tqdm(total=p.tiles, desc="USDU", unit="tile")

    image_mask = p.image_mask.convert("L")
    init_image = p.init_images[0]

    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
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
        crop_region, tile_size = expand_crop(
            crop_region,
            image_mask.width,
            image_mask.height,
            target_width,
            target_height,
        )
    else:
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(
            crop_region,
            image_mask.width,
            image_mask.height,
            target_width,
            target_height,
        )

    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    tiles = [img.crop(crop_region) for img in shared.batch]
    initial_tile_size = tiles[0].size

    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    positive_cropped = crop_cond(
        p.positive,
        crop_region,
        p.init_size,
        init_image.size,
        tile_size,
    )
    negative_cropped = crop_cond(
        p.negative,
        crop_region,
        p.init_size,
        init_image.size,
        tile_size,
    )

    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    samples = sample(
        p.model,
        p.seed,
        p.steps,
        p.cfg,
        p.sampler_name,
        p.scheduler,
        positive_cropped,
        negative_cropped,
        latent,
        p.denoise,
        p.custom_sampler,
        p.custom_sigmas,
    )

    if p.progress_bar_enabled:
        p.pbar.update(1)

    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        image_tile_only = Image.new("RGBA", init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        result = init_image.convert("RGBA")
        result.alpha_composite(image_tile_only)

        result = result.convert("RGB")

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, None)
    return processed


# --- UPSCALER WRAPPERS ------------------------------------------------------

class Upscaler:
    def upscale(self, img, scale, selected_model: str = None):
        if scale == 1.0:
            return img
        if shared.actual_upscaler is None:
            return img.resize((img.width * scale, img.height * scale), Image.Resampling.LANCZOS)
        if hasattr(ImageUpscaleWithModel, "execute"):
            (upscaled,) = ImageUpscaleWithModel.execute(shared.actual_upscaler, shared.batch_as_tensor)
        else:
            (upscaled,) = ImageUpscaleWithModel().upscale(shared.actual_upscaler, shared.batch_as_tensor)
        shared.batch = [tensor_to_pil(upscaled, i) for i in range(len(upscaled))]
        return shared.batch[0]


class UpscalerData:
    name = ""
    data_path = ""

    def __init__(self):
        self.scaler = Upscaler()

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
