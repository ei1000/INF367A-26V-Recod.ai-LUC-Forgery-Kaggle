import numpy as np
from scipy import ndimage

class PixelMapUtil:
    def __init__(
            self,
            gaussian_sigma: float = 0.5,
    ):
        self.gaussian_sigma = gaussian_sigma
    
    def get_gaussian_sigma(self) -> float:
        return self.gaussian_sigma
    
    def set_gaussian_sigma(self, sigma: float) -> None:
        self.gaussian_sigma = sigma

    def _get_edges(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        sobel_x = ndimage.sobel(img, 0)
        sobel_y = ndimage.sobel(img, 1)
        mag = np.hypot(sobel_x, sobel_y)
        return mag
    
    def get_edges(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return self._get_edges(img)
    
    def _gaussian_blur(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        gaussian = ndimage.gaussian_filter(img, self.gaussian_sigma)
        return gaussian
    
    def gaussian_blur(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return self._gaussian_blur(img)
    
    def opening(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_opening(img, structure=np.ones((3,3)))

    def closing(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_closing(img, structure=np.ones((5,5)))

    def fill_components(self, img: np.typing.ArrayLike | np.typing.NDArray) -> np.typing.NDArray:
        return ndimage.binary_fill_holes(img) # type: ignore
    
    def post_process_img(self, img: np.typing.NDArray) -> np.typing.NDArray:
        img = img.copy()
        
        # Gaussian blur first to smooth noise
        img_blurred = self._gaussian_blur(img)
        
        # Get edges from blurred image (no need to blur again)
        edges = self._get_edges(img_blurred)
        edges = self._gaussian_blur(edges)
        edges = edges > 0.5
        edges = self.fill_components(edges)
        
        # Add edges to original
        img = img + edges - 1
        
        # Invert, close, open, invert back
        img = 1 - img
        img = self.closing(img)
        img = self.opening(img)
        img = 1 - img
        
        return img

    def post_process_mask_probs(
        self,
        probs: np.typing.NDArray,
        threshold: float = 0.5,
        confident_threshold: float | None = 0.9,
        smooth_probabilities: bool = True,
        fill_holes: bool = True,
        apply_opening: bool = True,
        apply_closing: bool = True,
        keep_confident_seeded_components: bool = False,
    ) -> np.typing.NDArray:
        probs = np.asarray(probs, dtype=np.float32).copy()
        probs = np.clip(probs, 0.0, 1.0)
        if confident_threshold is None:
            confident_threshold = 0.9

        threshold_map = self._gaussian_blur(probs) if smooth_probabilities else probs
        mask = threshold_map >= threshold
        confident = probs >= confident_threshold
        mask = np.logical_or(mask, confident)

        if apply_closing:
            mask = self.closing(mask)
        if apply_opening:
            mask = self.opening(mask)
        if fill_holes:
            mask = self.fill_components(mask)

        if keep_confident_seeded_components:
            confident_mask = confident.astype(bool)
            labeled, num = ndimage.label(mask)
            if num > 0:
                keep = np.zeros_like(mask, dtype=bool)
                for component_id in range(1, num + 1):
                    component = labeled == component_id
                    if np.any(np.logical_and(component, confident_mask)):
                        keep |= component
                mask = keep

        return mask.astype(np.float32)
