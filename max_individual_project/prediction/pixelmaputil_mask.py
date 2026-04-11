# CREDIT: Main project pixelmapUtil.py
# This is a toned down version specificially for editing output masks, with some changes to the ordering to adress the specific issues of the architecture
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

    def remove_small_components(
        self,
        img: np.typing.ArrayLike | np.typing.NDArray,
        min_area: int,
    ) -> np.typing.NDArray:
        mask = np.asarray(img, dtype=bool)
        if min_area <= 1:
            return mask

        labeled, count = ndimage.label(mask)
        if count == 0:
            return mask

        component_sizes = np.bincount(labeled.ravel())
        keep = component_sizes >= int(min_area)
        keep[0] = False
        return keep[labeled]

    def keep_components_with_seed(
        self,
        img: np.typing.ArrayLike | np.typing.NDArray,
        seeds: np.typing.ArrayLike | np.typing.NDArray,
    ) -> np.typing.NDArray:
        mask = np.asarray(img, dtype=bool)
        seed_mask = np.asarray(seeds, dtype=bool)
        if not mask.any() or not seed_mask.any():
            return np.zeros_like(mask, dtype=bool)

        labeled, count = ndimage.label(mask)
        if count == 0:
            return np.zeros_like(mask, dtype=bool)

        seeded_labels = np.unique(labeled[seed_mask & mask])
        seeded_labels = seeded_labels[seeded_labels != 0]
        if seeded_labels.size == 0:
            return np.zeros_like(mask, dtype=bool)

        keep = np.zeros(count + 1, dtype=bool)
        keep[seeded_labels] = True
        return keep[labeled]


    def post_process_mask_probs(
        self,
        probs: np.typing.NDArray,
        threshold: float = 0.5,
        confident_threshold: float | None = None,
        min_component_area: int = 0,
        smooth_probabilities: bool = False,
        fill_holes: bool = True,
        apply_closing: bool = False,
    ) -> np.typing.NDArray:
        probs = np.asarray(probs, dtype=np.float32).copy()
        probs = np.clip(probs, 0.0, 1.0)

        score_map = self.gaussian_blur(probs) if smooth_probabilities else probs
        mask = score_map >= threshold

        mask = self.opening(mask)
        mask = self.remove_small_components(mask, min_area=min_component_area)

        if confident_threshold is not None:
            confident = probs >= confident_threshold
            mask = self.keep_components_with_seed(mask, confident)
        if apply_closing:
            mask = self.closing(mask)
        if fill_holes:
            mask = self.fill_components(mask)

        return mask.astype(np.uint8)
