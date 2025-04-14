import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class DeepRTEConfig:
    position_coords_dim: int = 2
    velocity_coords_dim: int = 2
    coeffs_fn_dim: int = 2
    num_basis_functions: int = 64
    basis_function_encoder_dim: int = 128
    num_basis_function_encoder_layers: int = 4
    green_function_encoder_dim: int = 128
    num_green_function_encoder_layers: int = 4
    num_scattering_layers: int = 2
    scattering_dim: int = 128
    num_heads: int = 8
    qkv_dim: int = 16
    optical_depth_dim: int = 16
    name: str = "boundary"
    subcollocation_size: int = 128
    # Normalization constant of dataset/model.
    normalization: float = 1.0
    # Where to load the parameters from.
    load_parameters_path: str = ""

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)
