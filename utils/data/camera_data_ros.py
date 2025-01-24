import numpy as np
import torch

from utils.dataset_processing import image


class CameraData:
    """
    Dataset wrapper for the camera data.
    """
    def __init__(self,
                 width=1280,
                 height=720,
                 output_size=224,
                 include_depth=True,
                 include_rgb=True,
                 fx=606.7401123046875,  # Focal length in x-axis
                 fy=606.6471557617188,  # Focal length in y-axis
                 ppx=640.8602294921875,  # Principal point x-coordinate
                 ppy=365.228515625# Principal point y-coordinate
                 ):
        """
        :param output_size: Image output size in pixels (square)
        :param include_depth: Whether depth image is included
        :param include_rgb: Whether RGB image is included
        """
        self.output_size = output_size
        self.include_depth = include_depth
        self.include_rgb = include_rgb
        self.fx = fx
        self.fy = fy
        self.ppx = ppx
        self.ppy = ppy
        
        
        if include_depth is False and include_rgb is False:
            raise ValueError('At least one of Depth or RGB must be specified.')

        left = (width - output_size) // 2
        #top = (height - output_size) // 2
        top = height - output_size
        right = (width + output_size) // 2
        #bottom = (height + output_size) // 2
        bottom = height
        self.bottom_right = (bottom, right)
        self.top_left = (top, left)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_depth(self, img):
        depth_img = image.Image(img)
        depth_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        depth_img.normalise()
        depth_img.resize((self.output_size, self.output_size))
        #depth_img.img = depth_img.img.transpose((2, 0, 1))
        return depth_img.img

    def get_rgb(self, img, norm=True):
        rgb_img = image.Image(img)
        rgb_img.crop(bottom_right=self.bottom_right, top_left=self.top_left)
        rgb_img.resize((self.output_size, self.output_size))
        if norm:
                rgb_img.normalise()
                rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

  
    def get_data(self, rgb=None, depth=None):
        depth_img = None
        rgb_img = None

        if self.include_depth:
            depth_img = self.get_depth(img=depth)
            depth_img = np.expand_dims(depth_img, axis=0)  # (1, H, W)

        if self.include_rgb:
            rgb_img = self.get_rgb(img=rgb)

        if self.include_depth and self.include_rgb:
            # Debugging statements to check dimensions
            print("Depth image shape:", depth_img.shape)
            print("RGB image shape:", rgb_img.shape)

            # Ensure dimensions match before concatenation
            if depth_img.shape[1:] != rgb_img.shape[1:]:
                raise ValueError(
                    f"Dimension mismatch: depth {depth_img.shape}, rgb {rgb_img.shape}"
                )

            x = self.numpy_to_torch(
                np.concatenate(
                    (depth_img, rgb_img), axis=0  # Concatenate along channel axis
                )
            )
        elif self.include_depth:
            x = self.numpy_to_torch(depth_img)
        elif self.include_rgb:
            x = self.numpy_to_torch(rgb_img)

        return x, depth_img, rgb_img
