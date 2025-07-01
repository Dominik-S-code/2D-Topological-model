import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import Image
import io
import time
import os
from os.path import splitext
from typing import Union
from argparse import ArgumentParser
import torch

from Input import Input
from collections.abc import Iterable

from Generators.Initializer import Initializer
from Generators.Initializer_cpu import Initializer_cpu

from Generators.Generator import Generator
from Generators.Generator_cpu import Generator_cpu
from Generators.Generator_torchBatch import Generator_torchBatch
from Generators.Generator_torch import Generator_torch



class Scrambling:
    def __init__(self, initializer: Initializer, generator: Generator):
        self.initializer = initializer
        self.generator = generator

    def generate_image(self, input: Input, t: Union[float, Iterable[float]], filename: str = None):
        """
        Generate an image/batch of images of the scrambling results.
        """
        start = time.time()
        generator_data = self.initializer.initialize(input)
        print ("Initialization complete.", "Time:", time.time() - start, "seconds")
        start = time.time()
        if not isinstance(t, Iterable):
            image = self.generator.generate_frame(generator_data, t)
            print("Image generation complete.", "Time:", time.time() - start, "seconds")
            self._save_image(0, image, filename,t)
        else:
            images = self.generator.generate_batch(generator_data, t)
            print("Batch generation complete.", "Time:", time.time() - start, "seconds")
            os.makedirs(filename, exist_ok=True)
            for i, frame in enumerate(images):
                self._save_image(i, frame, f'{filename}img_{i}',t[i])
    
    def generate_video(self, input: Input, range: tuple[float, float], fps: float, filename: str = None):
        """
        Generate a video of the scrambling results.
        """
        start = time.time()
        generator_data = self.initializer.initialize(input)
        print ("Initialization complete.", "Time:", time.time() - start, "seconds")
        start = time.time()
        t = np.arange(range[0], range[1], range[2])
        images = self.generator.generate_batch(generator_data, t)
        print("Video generation complete.", "Time:", time.time() - start, "seconds")
        self._save_gif(images, fps, filename)

    def _save_gif(self, frames, fps, filename):
        new_frames = []
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['white', 'blue'], 256)

        for frame in frames:
            fig, ax = plt.subplots()
            ax.imshow(frame, interpolation='nearest', cmap=cmap)
            ax.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=400, bbox_inches='tight')
            buf.seek(0)

            new_frames.append(Image.open(buf))
            plt.close(fig)

        new_frames[0].save(
            f'{filename}.gif',
            save_all=True,
            append_images=new_frames[1:],
            duration=1000/fps,  
            loop=0  
        )
        print("GIF saved as", filename + ".gif")


    def _save_image(self, num, frame, filename,t):
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['white','blue'], 256)
        plt.imshow(frame, interpolation='nearest', cmap=cmap)
        plt.axis('off')
        plt.text(2,2,f't={t}', fontsize=16)
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Image saved as", filename + ".png")


def gather_args():
    parser = ArgumentParser(description="Generate scrambling results.")
    mode_subparsers = parser.add_subparsers(dest='mode', required=True, help="Mode of operation: 'image' or 'video'.")
    image_parser = mode_subparsers.add_parser('image', help="Generate a single image or a series of images.")
    image_parser.add_argument('--system_size', '-n', type=int, help="System size.", default=10)
    image_parser.add_argument('--time', '-t', type=float, help="Time at which to generate the image.", default=5.0)
    image_parser.add_argument('--filename', type=str, help="Filename to save the image to.", default=None)
    image_parser.add_argument('--range', '-r', type=float, nargs=3, help="Range of time: start stop jump.", default=None)

    video_parser = mode_subparsers.add_parser('video', help="Generate a video.")
    video_parser.add_argument('--system_size', '-n', type=int, help="System size.", default=10)
    video_parser.add_argument('--range', '-r', type=float, nargs=3, help="Range of time: start stop jump.", default=(0.0, 10.0, 1.0))
    video_parser.add_argument('--fps', type=float, help="Frames per second of the video.", default=15.0)
    video_parser.add_argument('--filename', type=str, help="Filename to save the video to.", default=None)

    parser.add_argument('--generator', '-g', type=str, choices=['cpu', 'torch'], help="Acceleration mode: 'cpu' or 'torch'.", default='torch')
    parser.add_argument('--device', '-d', type=str, help="With torch generator. Device to use (e.g. 'cuda' or 'cpu').", default=None)
    parser.add_argument('--torch_batch', '-b',  action='store_true', help="Use torch batch generator. (Memory heavy)")

    return parser.parse_args()

def get_generator(args):
    if args.generator == 'cpu':
            return Generator_cpu()
    elif args.generator == 'torch':
        device = args.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device:", device)
        if args.torch_batch:
            return Generator_torchBatch(device)
        else:
            return Generator_torch(device)
    else:
        raise ValueError("Invalid generator. Use 'cpu' or 'torch'.")
    
def range_to_text(range):
    range_start, range_end, range_jump = range
    range_start = f"{range_start:.2f}".replace('.', 'p')
    range_end = f"{range_end:.2f}".replace('.', 'p')
    range_jump = f"{range_jump:.2f}".replace('.', 'p')
    return range_start, range_end, range_jump

def get_filename(args):
    if args.filename is None:
        if args.mode == 'image':
            if args.range is not None:
                range_start, range_end, range_jump = range_to_text(args.range)
                return f"./out/batch_n{args.system_size}_range{range_start}_{range_end}_{range_jump}/"
            else:
                t = f"{args.time:.2f}".replace('.', 'p')
                return f"./out/img_n{args.system_size}_t{t}"
        elif args.mode == 'video':
            range_start, range_end, range_jump = range_to_text(args.range)
            return f"./out/vid_n{args.system_size}_range{range_start}_{range_end}_{range_jump}_fps{args.fps}"
    else:
        return splitext(args.filename)[0]

    
if __name__ == "__main__":
    args = gather_args()

    input = Input(args.system_size)
    generator = get_generator(args)
    initializer = Initializer_cpu()
    scrambling = Scrambling(initializer, generator)
    filename = get_filename(args)

    if args.mode == 'image':
        if args.range is not None:
            range = np.arange(args.range[0], args.range[1], args.range[2])
            print("Range:", range)
            scrambling.generate_image(input, range, filename)
        else:
            scrambling.generate_image(input, args.time, filename)
    elif args.mode == 'video':
        scrambling.generate_video(input, args.range, args.fps, filename)
    else:
        raise ValueError("Invalid mode. Use 'image' or 'video'.")

    
        
        
