import torch
from typing import Tuple, List, Optional, Union

class GridTiling:
    """
    Grid Tiling Operator.
    Extracts multiple uniform-sized tiles from an input tensor based on specified top-left coordinates.
    """

    def __init__(self,
                 grid_coords: List[Tuple[int, int]],
                 tile_shape: Tuple[int, int],
                 tiling_axes: Tuple[int, int],
                 axis: int = 1,
                 permute: Optional[Tuple[int, ...]] = None,
                 norm_factor: Optional[float] = None):
        """
        Initialize Gird Tiling Operator.

        Args:
            grid_coords (List[Tuple[int, int]]): Top-left coordinates (x, y) of each tile.
            tile_shape (Tuple[int, int]): Size of all tiles (height, width).
            tiling_axes (Tuple[int, int]): Which two axes in the input tensor correspond to H and W.
            axis (int): The axis at which the new 'N_tiles' dimension will be inserted.
            permute (Optional[Tuple[int, ...]]): Permutation to apply to each tile *before* inserting the N axis.
            norm_factor (Optional[float]): Normalization factor. If None, it's inferred from input dtype.
        """
        self._grid_coords = grid_coords
        self._tile_shape = tile_shape
        self._tiling_axes = tiling_axes
        self._axis = axis
        self._permute = permute
        self._norm_factor = norm_factor
        
        # Calculate minimum required width and height
        self._min_w = max([x + tile_shape[1] for x, y in grid_coords]) if grid_coords else 0
        self._min_h = max([y + tile_shape[0] for x, y in grid_coords]) if grid_coords else 0
        
    @property
    def grid_coords(self) -> List[Tuple[int, int]]:
        """Returns the grid top-left coordinates."""
        return self._grid_coords
        
    @property
    def tile_shape(self) -> Tuple[int, int]:
        """Returns the shape of each tile (H, W)."""
        return self._tile_shape
        
    def process(self,
                input_tensor: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                tiling_axes: Optional[Tuple[int, int]] = None,
                axis: Optional[int] = None,
                permute: Optional[Tuple[int, ...]] = None,
                norm_factor: Optional[float] = None) -> torch.Tensor:
        """
        Process the input tensor, extracting tiles and formatting them.
        
        Args:
            input_tensor (torch.Tensor): The input tensor to extract tiles from.
            out (Optional[torch.Tensor]): Pre-allocated output tensor. Must match expected shape.
            tiling_axes (Optional[Tuple[int, int]]): Override initialization parameter.
            axis (Optional[int]): Override initialization parameter.
            permute (Optional[Tuple[int, ...]]): Override initialization parameter.
            norm_factor (Optional[float]): Override initialization parameter.
            
        Returns:
            torch.Tensor: The tensor containing the extracted tiles.
        """
        tiling_axes = tiling_axes if tiling_axes is not None else self._tiling_axes
        axis = axis if axis is not None else self._axis
        permute = permute if permute is not None else self._permute
        norm_factor = norm_factor if norm_factor is not None else self._norm_factor
        
        h_axis, w_axis = tiling_axes
        
        # Handle negative axes
        ndim = input_tensor.ndim
        h_axis = h_axis if h_axis >= 0 else ndim + h_axis
        w_axis = w_axis if w_axis >= 0 else ndim + w_axis
        
        if input_tensor.shape[h_axis] < self._min_h or input_tensor.shape[w_axis] < self._min_w:
            raise ValueError(f"Input tensor shape {input_tensor.shape} is too small for tiling. "
                             f"Required min HxW: {self._min_h}x{self._min_w}")
                             
        if norm_factor is None:
            if input_tensor.dtype == torch.uint8:
                norm_factor = 255.0
            elif input_tensor.dtype in (torch.int16, torch.uint16, torch.int32):
                norm_factor = 65535.0
            else:
                norm_factor = 1.0
                
        N = len(self._grid_coords)
        H_t, W_t = self._tile_shape
        
        # Determine intermediate shape of a tile before adding N axis
        sliced_shape = list(input_tensor.shape)
        sliced_shape[h_axis] = H_t
        sliced_shape[w_axis] = W_t
        
        # Apply permute if specified
        if permute:
            tile_shape_post_permute = [sliced_shape[i] for i in permute]
        else:
            tile_shape_post_permute = sliced_shape
            
        # Final output shape with N inserted
        out_shape = list(tile_shape_post_permute)
        
        # Handle negative axis for insert
        out_ndim = len(out_shape) + 1
        insert_axis = axis if axis >= 0 else out_ndim + axis
        out_shape.insert(insert_axis, N)
        
        if out is None:
            out = torch.empty(out_shape, dtype=torch.float32, device=input_tensor.device)
        else:
            if out.numel() != torch.prod(torch.tensor(out_shape)):
                 raise ValueError(f"Output tensor size mismatch. Expected {torch.prod(torch.tensor(out_shape))} elements, got {out.numel()}")
            out = out.view(out_shape)
            
        for i, (tx, ty) in enumerate(self._grid_coords):
            slices = [slice(None)] * ndim
            slices[h_axis] = slice(ty, ty + H_t)
            slices[w_axis] = slice(tx, tx + W_t)
            
            patch = input_tensor[tuple(slices)]
            
            if permute:
                patch = patch.permute(*permute)
                
            out_slices = [slice(None)] * out.ndim
            out_slices[insert_axis] = i
            
            if norm_factor != 1.0 or patch.dtype != out.dtype:
                out[tuple(out_slices)] = (patch.float() / norm_factor).to(out.dtype)
            else:
                out[tuple(out_slices)] = patch            
        return out
        
    def to_global_coord(self, local_coords: Union[Tuple[int, float, float], List[Tuple[int, float, float]]]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """
        Convert local tile coordinate(s) to global coordinate(s).
        
        Args:
            local_coords: A single Tuple (tile_index, x, y) or a List of such Tuples.
            
        Returns:
            A single Tuple (global_x, global_y) or a List of such Tuples.
        """
        is_single = isinstance(local_coords, tuple)
        coords = [local_coords] if is_single else local_coords
        
        res = []
        for tile_idx, x, y in coords:
            tx, ty = self._grid_coords[tile_idx]
            res.append((x + tx, y + ty))
            
        return res[0] if is_single else res
        
    def to_local_coord(self, global_coords: Union[Tuple[float, float], List[Tuple[float, float]]]) -> Union[Optional[Tuple[int, float, float]], List[Optional[Tuple[int, float, float]]]]:
        """
        Convert global coordinate(s) to local tile coordinate(s).
        Finds the first tile that contains the global coordinate.
        
        Args:
            global_coords: A single Tuple (global_x, global_y) or a List of such Tuples.
            
        Returns:
            A single Tuple (tile_index, local_x, local_y) or a List of such Tuples.
            If a global coordinate is not within any tile's boundaries, returns None for that coordinate.
        """
        is_single = isinstance(global_coords, tuple)
        coords = [global_coords] if is_single else global_coords
        
        H_t, W_t = self._tile_shape
        res = []
        for x, y in coords:
            match = None
            for i, (tx, ty) in enumerate(self._grid_coords):
                if tx <= x < tx + W_t and ty <= y < ty + H_t:
                    match = (i, x - tx, y - ty)
                    break
            res.append(match)
            
        return res[0] if is_single else res
