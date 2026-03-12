import torch
import torch.nn.functional as F


class PixelPropagator:
    def __init__(self, image: torch.Tensor, cnn_features, zernike_features):
        self.image = image
        self.cnn_features = cnn_features
        self.zernike_features = zernike_features
        self.max_fraction = 1

    '''
    i) Initialization layer: Generate random offsets for each pixel

    Does this through torch for GPU efficiency.
    Does not consider trivial offset (0, 0)

    Image:
    Datatype: torch.Tensor
    Dtype: torch.float32
    '''
    def generate_random_offsets(self):
        _, H, W = self.image.size()

        x_max_offset = int(W * self.max_fraction)
        y_max_offset = int(H * self.max_fraction)

        # Create a grid of original coordinates
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.image.device),
            torch.arange(W, device=self.image.device),
            indexing='ij',
        )

        # Generate all possible offset pairs excluding (0,0)
        x_offsets_range = torch.arange(-x_max_offset, x_max_offset + 1, device=self.image.device)
        y_offsets_range = torch.arange(-y_max_offset, y_max_offset + 1, device=self.image.device)
        xx, yy = torch.meshgrid(x_offsets_range, y_offsets_range, indexing='ij')
        all_offsets = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Remove (0,0)
        all_offsets = all_offsets[(all_offsets[:, 0] != 0) | (all_offsets[:, 1] != 0)]

        # Sample a random offset for each pixel
        indices = torch.randint(0, all_offsets.shape[0], (H, W), device=self.image.device)
        offsets = all_offsets[indices]

        x_offsets = offsets[:, :, 0]
        y_offsets = offsets[:, :, 1]

        # Apply offsets and clip to image boundaries
        new_x = (x_grid + x_offsets).clamp(0, W - 1)
        new_y = (y_grid + y_offsets).clamp(0, H - 1)

        return new_x, new_y
    
    '''
    ii) Propagation layer: Propagate pixel offsets to neighbouring pixels.
    Utilizes propagation block and random search block
    '''
    def propagation_layer(self, iters=5):
        x, y = self.generate_random_offsets()
        for _ in range(iters):
            basic_candidates = x, y
            propagation_candidates = self.propagation_block(x, y)
            random_candidates = self.random_search_block(x, y)
        # TODO: return max of propagation and random search
        return basic_candidates

    '''
    Propagate to every pixel from surrounding pixels
    Uses a combination of zero order and first order offset

    Every (x, y) entry takes offsets from surrounding pixels as candidates.
    Then directly evaluate all candidates - evaluate them to avoid memory usage.
    Return the best to be checked against the random search candidates.
    '''
    def propagation_block(self, x_entries: torch.Tensor, y_entries: torch.Tensor) -> torch.Tensor:
        H, W = x_entries.shape

        # Padding: To include borders
        pad_zero = 1   # for 3x3 windows 
        pad_first = 2  # for 5x5 windows

        x_entries_f = x_entries.float()
        y_entries_f = y_entries.float()

        x_zero = F.unfold(
            x_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=pad_zero,
        ).view(1, 9, H, W)
        y_zero = F.unfold(
            y_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=pad_zero,
        ).view(1, 9, H, W)

        # center of zero-order windows (index 4 in 3x3)
        x_center = x_zero[:, 4]
        y_center = y_zero[:, 4]

        # 4 adjacent neighbors in zero-order (up, left, right, down)
        adj_idx = [1, 3, 5, 7]
        adj_x = x_zero[:, adj_idx].permute(2, 3, 1, 0).squeeze(-1)
        adj_y = y_zero[:, adj_idx].permute(2, 3, 1, 0).squeeze(-1)

        # sliding windows for first-order (5x5)
        x_first = F.unfold(
            x_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            padding=pad_first,
        ).view(1, 25, H, W)
        y_first = F.unfold(
            y_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            padding=pad_first,
        ).view(1, 25, H, W)

        # 8 first-order neighbors (corners + midpoints)
        first_idx = [0, 2, 4, 10, 14, 20, 22, 24]
        x_first = x_first[:, first_idx].permute(2, 3, 1, 0).squeeze(-1)
        y_first = y_first[:, first_idx].permute(2, 3, 1, 0).squeeze(-1)

        # first-order candidates: 2*zero_order_center - first_order
        first_x = 2 * x_center.permute(1, 2, 0) - x_first
        first_y = 2 * y_center.permute(1, 2, 0) - y_first

        # stack adjacent neighbors and first-order candidates
        adj = torch.stack((adj_x, adj_y), dim=-1)
        first = torch.stack((first_x, first_y), dim=-1)
        candidates = torch.cat((adj, first), dim=2)

        return candidates

    '''
    Generate k candidates within the radius for each pixel to test similarity to

    '''
    def random_search_block(self, x_entries, y_entries, radius=50, k=4):
        H, W = x_entries.shape
        x_entries_f = x_entries.float()
        y_entries_f = y_entries.float()

        # Sample k random offsets per pixel within the radius (avoid large unfold buffers)
        dx = torch.randint(-radius, radius + 1, (H, W, k), device=self.image.device)
        dy = torch.randint(-radius, radius + 1, (H, W, k), device=self.image.device)

        # Avoid the trivial (0,0) offset by nudging dy when both are zero
        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy = dy.clone()
            dy[zero_mask] = 1

        x_candidates = (x_entries_f.unsqueeze(-1) + dx).clamp(0, W - 1)
        y_candidates = (y_entries_f.unsqueeze(-1) + dy).clamp(0, H - 1)
        candidates = torch.stack((x_candidates, y_candidates), dim=-1)
        return candidates
    
    def evaluate(self, candidates):

        # Use summing formula from paper
        # Clip to make sure all are still valid offsets

        
        raise NotImplementedError("PixelPropagator.evaluate is not implemented yet.")