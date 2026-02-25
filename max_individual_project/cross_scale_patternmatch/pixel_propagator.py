import numpy as np
import torch

class PixelPropagator:
    def __init__(self, image: torch.Tensor):
        self.image = image
        self.max_fraction = 1

    '''
    i) Initialization layer: Generate random offsets for each pixel

    Does this through numpy for efficiency.
    Does not consider trivial offset (0, 0)

    Image:
    Datatype: torch.Tensor
    Dtype: torch.float32
    '''
    def generate_random_offsets(self, ):
        C, W, H = self.image.size()

        x_max_offset = int(W * self.max_fraction)
        y_max_offset = int(H * self.max_fraction)

        # Create a grid of original coordinates
        x_grid, y_grid = np.meshgrid(np.arange(W), np.arange(H), indexing='ij')

        # Generate all possible offset pairs excluding (0,0)
        x_offsets_range = np.arange(-x_max_offset, x_max_offset + 1)
        y_offsets_range = np.arange(-y_max_offset, y_max_offset + 1)
        xx, yy = np.meshgrid(x_offsets_range, y_offsets_range, indexing='ij')
        all_offsets = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Remove (0,0)
        all_offsets = all_offsets[(all_offsets[:,0] != 0) | (all_offsets[:,1] != 0)]

        # Sample a random offset for each pixel
        indices = np.random.randint(0, all_offsets.shape[0], size=(W, H))
        offsets = all_offsets[indices]

        x_offsets = offsets[:,:,0]
        y_offsets = offsets[:,:,1]

        # Apply offsets and clip to image boundaries
        new_x = np.clip(x_grid + x_offsets, 0, W - 1)
        new_y = np.clip(y_grid + y_offsets, 0, H - 1)

        return new_x, new_y
    
    '''
    ii) Propagation layer: Propagate pixel offsets to neighbouring pixels.
    Utilizes propagation block and random search block
    '''
    def propagation_layer(self):
        x, y = self.generate_random_offsets(self.image)
        grid = self.propagation_block(x, y)
        return grid

    '''
    Propagate to every pixel from surrounding pixels
    Uses a combination of zero order and first order offset

    Every (x, y) entry takes offsets from surrounding pixels
    '''
    def propagation_block(self, x_entries: np.ndarray, y_entries: np.ndarray) -> np.ndarray:
        new_values = np.zeros((x_entries.shape[0], x_entries.shape[1], 2))

        zero_order_offsets = [
            (-1, 1), (-1, 0), (-1, -1),
            (0, 1), (0, -1),
            (1, 1), (1, 0), (1, -1),
        ]
        first_order_offsets = [
            (-2, -2), (-2, 0), (-2, 2),
            (0, -2), (0, 2),
            (2, -2), (2, 0), (2, 2)
        ]

        W, H = x_entries.shape

        for i in range(W):
            for j in range(H):
                zero_order = np.zeros(2)
                for x_offset, y_offset in zero_order_offsets:
                    ni, nj = i + x_offset, j + y_offset
                    if 0 <= ni < W and 0 <= nj < H:
                        zero_order[0] += x_entries[ni, nj]
                        zero_order[1] += y_entries[ni, nj]
                
                first_order = np.zeros(2)
                for x_offset, y_offset in first_order_offsets:
                    ni, nj = i + x_offset, j + y_offset
                    if 0 <= ni < W and 0 <= nj < H:
                        first_order[0] += x_entries[ni, nj]
                        first_order[1] += y_entries[ni, nj]
                
                # Average to restrict values - not mentioned in paper but makes sense.
                # If not, most of our generated offsets are invalid. Now most are valid, we just have to clip
                zero_order //= len(zero_order_offsets)
                first_order //= len(first_order_offsets)
                
                # Use summing formula from paper
                # Clip to make sure all are still valid offsets
                new_values[i, j, 0] = np.clip(2*zero_order[0] - first_order[0], 0, W-1)
                new_values[i, j, 1] = np.clip(2*zero_order[1] - first_order[1], 0, H-1)
        
        return new_values


    '''
    Generate k candidates within the radius to test similarity to.
    '''
    def random_search_block(self, x_entries, y_entries, radius=50, k=4):
        C, W, H = self.image.size()

        candidate_x = np.zeros(k)
        candidate_y = np.zeros(k)

        # Create candidates for each pixel
        for i in range(W):
            for j in range(H):
                x_ranges = np.array((np.min(0, i - radius), np.max(W, i + radius)))
                y_ranges = np.array((np.min(0, j - radius), np.max(H, j + radius)))
                
                x = np.random.choice(x_ranges[0], x_ranges[1])
                y = np.random.choice(y_ranges[0], y_ranges[1])
        