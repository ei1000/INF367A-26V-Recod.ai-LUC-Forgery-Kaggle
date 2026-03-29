import numpy as np
import torch

class MultiScaleDLF:
    def __init__(self, image: torch.Tensor, cnn_offsets: torch.Tensor, radiuses: np.ndarray | None): # zernike_offsets likely not used
        self.image = image
        self.cnn_offsets = cnn_offsets

        if radiuses is None:
            radiuses = np.array([7, 9, 11])
        
        self.radiuses = radiuses

    '''
    Perform multi-scale dense linear fitting based on P + delta_P ≈ PA, B = A - I

    Solves the linear regression problem using closed form solution:
    B* = (P'P)**-1 P' * delta_P

    Solves for both x and y offsets and combines errors.

    Multi scale as it uses all radiuses in the self.radius array to account for different scalings.

    @return errors for each scale
    '''
    def predict(self):
        pass