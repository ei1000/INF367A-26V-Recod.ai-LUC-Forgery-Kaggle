# CREDIT: Main project pixelmapUtil.py
# This is a toned down version specificially for editing output masks
import numpy as np
from scipy import ndimage

class MaskUtil:
    def __init__(
            self,
            gaussian_sigma: float = 0.5,
    ):
        self.gaussian_sigma = gaussian_sigma
    
    def gaussian_blur(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        gaussian = ndimage.gaussian_filter(img, self.gaussian_sigma)
        return gaussian


    def opening(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_opening(img, structure=np.ones((3,3)))

    def closing(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_closing(img, structure=np.ones((5,5)))

    def fill_components(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_fill_holes(img) # type: ignore
    

    def post_process_mask_probs(
        self,
        probs: np.typing.NDArray,
        threshold: float = 0.5,
        confident_threshold: float = 0.8,
    ) -> np.typing.NDArray:
        probs = np.asarray(probs, dtype=np.float32).copy()
        probs = np.clip(probs, 0.0, 1.0)

        smooth = self.gaussian_blur(probs)
        mask = smooth >= threshold
        confident = probs >= confident_threshold
        mask = np.logical_or(mask, confident)

        mask = self.closing(mask)
        mask = self.opening(mask)
        mask = self.fill_components(mask)
        return mask.astype(np.uint8)
