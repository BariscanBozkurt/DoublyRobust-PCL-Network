import torch
from torch.utils import data
from typing import Optional, Tuple, List, Union


class BackdoorDataset(data.Dataset):
    """
    PyTorch Dataset for causal inference with backdoor adjustment.

    Supports observed outcomes, treatment effects, and optional data
    transformations.

    Parameters
    ----------
    data_tuple : tuple
        Tuple containing data arrays: (A, Y, U) or (A, Y, U, TE), where
        A : treatment features
        Y : observed outcome
        U : confounders
        TE : optional treatment effect (used if target_type="TE")
    target_type : str, default="Y"
        Which target to use:
        - "Y": use observed outcome
        - "TE": use treatment effect
        - None: no target
    transformers : list, optional
        List of transformers for (A, Y, U). Each transformer should implement
        `fit_transform()`. Default is identity transformers.
    device : str, default="cpu"
        Torch device on which to store the tensors.
    """

    def __init__(
        self,
        data_tuple: Union[Tuple, List],
        target_type: str = "Y",
        transformers: Optional[List] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.target_type = target_type
        self.device = device

        # Default transformers = identity
        if transformers is None:
            transformers = [
                TorchIdentityTransformer(),
                TorchIdentityTransformer(),
                TorchIdentityTransformer(),
            ]
        self.transformers = transformers

        # Unpack data tuple
        if len(data_tuple) == 3:
            A, Y, U = data_tuple
            TE = None
        elif len(data_tuple) == 4:
            A, Y, U, TE = data_tuple
        else:
            raise ValueError("data_tuple must have 3 or 4 elements")

        self.size = U.shape[0]

        # Apply transformations
        A = self.transformers[0].fit_transform(A)
        if Y is not None:
            Y = self.transformers[1].fit_transform(Y)
        U = self.transformers[2].fit_transform(U)
        if TE is not None:
            TE = self.transformers[1].fit_transform(TE)  # reuse Y transformer

        # Convert to torch tensors
        self.A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        self.U = torch.as_tensor(U, dtype=torch.float32, device=self.device)
        self.Y = None if Y is None else torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        self.TE = None if TE is None else torch.as_tensor(TE, dtype=torch.float32, device=self.device)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple
            If target_type is "Y" or "TE": (A[idx], target[idx], U[idx])
            If target_type is None: (A[idx], U[idx])
        """
        if self.target_type == "Y":
            target = self.Y
        elif self.target_type == "TE":
            target = self.TE
        else:
            target = None

        if target is not None:
            return self.A[idx], target[idx], self.U[idx]
        else:
            return self.A[idx], self.U[idx]


class ProxyDataset(data.Dataset):
    """
    PyTorch Dataset for Proxy Causal Learning with an optional backdoor variable X.

    Parameters
    ----------
    data_tuple : tuple or list
        Tuple containing data arrays in one of the following formats:
        - (A, Y, Z, W)    -> Without high-dimensional covariates
        - (A, Y, Z, W, X) -> With high-dimensional covariates (backdoor)
    """

    def __init__(
        self,
        data_tuple: Union[Tuple, List],
        dens_ratio: Optional[Union[torch.Tensor, list, object]] = None,
        dens_ratio_transformer: Optional[object] = None,
        transformers: Optional[List] = None,
        device: str = "cpu",
        is_train_set: bool = True
    ):
        super().__init__()
        self.device = device
        self.is_train_set = is_train_set

        if transformers is None:
            from torch_utils.scalers import TorchIdentityTransformer
            transformers = [TorchIdentityTransformer() for _ in range(5)]

        self.transformers = transformers
        self.dens_ratio_transformer = dens_ratio_transformer

        def _apply_transform(data, scaler):
            if data is None or scaler is None:
                return data
            is_fitted = scaler.is_fitted if hasattr(scaler, 'is_fitted') else False
            if self.is_train_set and not is_fitted:
                return scaler.fit_transform(data)
            return scaler.transform(data)

        # --- Unpack Data Tuple ---
        A, Y, Z, W, X = None, None, None, None, None

        if len(data_tuple) == 4:
            A, Y, Z, W = data_tuple
        elif len(data_tuple) == 5:
            A, Y, Z, W, X = data_tuple
        else:
            raise ValueError("data_tuple must have exactly 4 or 5 elements.")

        self.size = A.shape[0]

        # --- Apply Standard Transformations ---
        A = _apply_transform(A, self.transformers[0])
        Y = _apply_transform(Y, self.transformers[1])
        Z = _apply_transform(Z, self.transformers[2])
        W = _apply_transform(W, self.transformers[3])

        # FIXED: Only access index 4 if X is present AND the list is long enough
        if X is not None:
            x_transformer = self.transformers[4] if len(self.transformers) > 4 else None
            X = _apply_transform(X, x_transformer)

        # --- Apply Density Ratio Transformation ---
        if dens_ratio is not None:
            if hasattr(dens_ratio, "ndim") and dens_ratio.ndim == 1:
                if isinstance(dens_ratio, torch.Tensor):
                    dens_ratio = dens_ratio.view(-1, 1)
                else:
                    import numpy as np
                    dens_ratio = np.array(dens_ratio).reshape(-1, 1)

            dens_ratio = _apply_transform(dens_ratio, self.dens_ratio_transformer)
            self.dens_ratio = torch.as_tensor(dens_ratio, dtype=torch.float32, device=self.device)
        else:
            self.dens_ratio = None

        # --- Convert Remaining Data to Tensors ---
        self.A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        self.Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        self.Z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        self.W = torch.as_tensor(W, dtype=torch.float32, device=self.device)
        self.X = None if X is None else torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # --- Move Transformers to Device ---
        for i in range(len(self.transformers)):
            if hasattr(self.transformers[i], 'to'):
                self.transformers[i] = self.transformers[i].to(device)

        if hasattr(self.dens_ratio_transformer, 'to'):
            self.dens_ratio_transformer = self.dens_ratio_transformer.to(device)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        out = (self.A[idx], self.Y[idx], self.Z[idx], self.W[idx])
        if self.X is not None:
            out += (self.X[idx],)
        if self.dens_ratio is not None:
            out += (self.dens_ratio[idx],)
        return out


class HeterogeneousProxyDataset(data.Dataset):
    """
    PyTorch Dataset for Heterogeneous Proxy Causal Learning.
    Supports a heterogeneity variable V and an optional backdoor variable X.

    Parameters
    ----------
    data_tuple : tuple or list
        Tuple containing data arrays in one of the following formats:
        - (A, Y, Z, W, V)    -> Without high-dimensional backdoor covariates
        - (A, Y, Z, W, V, X) -> With high-dimensional backdoor covariates
    """

    def __init__(
        self,
        data_tuple: Union[Tuple, List],
        dens_ratio: Optional[Union[torch.Tensor, list, object]] = None,
        dens_ratio_transformer: Optional[object] = None,
        transformers: Optional[List] = None,
        device: str = "cpu",
        is_train_set: bool = True
    ):
        super().__init__()
        self.device = device
        self.is_train_set = is_train_set

        # Initialize transformers if not provided (6 slots: A, Y, Z, W, V, X)
        if transformers is None:
            from torch_utils.scalers import TorchIdentityTransformer
            transformers = [TorchIdentityTransformer() for _ in range(6)]

        self.transformers = transformers
        self.dens_ratio_transformer = dens_ratio_transformer

        def _apply_transform(data, scaler):
            if data is None or scaler is None:
                return data
            is_fitted = scaler.is_fitted if hasattr(scaler, 'is_fitted') else False
            if self.is_train_set and not is_fitted:
                return scaler.fit_transform(data)
            return scaler.transform(data)

        # --- Unpack Data Tuple ---
        A, Y, Z, W, V, X = None, None, None, None, None, None

        if len(data_tuple) == 5:
            A, Y, Z, W, V = data_tuple
        elif len(data_tuple) == 6:
            A, Y, Z, W, V, X = data_tuple
        else:
            raise ValueError("data_tuple must have exactly 5 (A,Y,Z,W,V) or 6 (A,Y,Z,W,V,X) elements.")

        self.size = A.shape[0]

        # --- Apply Transformations ---
        # Indices: 0:A, 1:Y, 2:Z, 3:W, 4:V, 5:X
        A = _apply_transform(A, self.transformers[0])
        Y = _apply_transform(Y, self.transformers[1])
        Z = _apply_transform(Z, self.transformers[2])
        W = _apply_transform(W, self.transformers[3])
        V = _apply_transform(V, self.transformers[4])

        if X is not None:
            x_transformer = self.transformers[5] if len(self.transformers) > 5 else None
            X = _apply_transform(X, x_transformer)

        # --- Apply Density Ratio Transformation ---
        if dens_ratio is not None:
            if hasattr(dens_ratio, "ndim") and dens_ratio.ndim == 1:
                if isinstance(dens_ratio, torch.Tensor):
                    dens_ratio = dens_ratio.view(-1, 1)
                else:
                    import numpy as np
                    dens_ratio = np.array(dens_ratio).reshape(-1, 1)

            dens_ratio = _apply_transform(dens_ratio, self.dens_ratio_transformer)
            self.dens_ratio = torch.as_tensor(dens_ratio, dtype=torch.float32, device=self.device)
        else:
            self.dens_ratio = None

        # --- Convert Data to Tensors ---
        self.A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        self.Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        self.Z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)
        self.W = torch.as_tensor(W, dtype=torch.float32, device=self.device)
        self.V = torch.as_tensor(V, dtype=torch.float32, device=self.device)
        self.X = None if X is None else torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # --- Move Transformers to Device ---
        for transformer in self.transformers:
            if hasattr(transformer, 'to'):
                transformer.to(device)

        if self.dens_ratio_transformer and hasattr(self.dens_ratio_transformer, 'to'):
            self.dens_ratio_transformer.to(device)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        # Always return the core 5 variables for the heterogeneous case
        out = (self.A[idx], self.Y[idx], self.Z[idx], self.W[idx], self.V[idx])

        # Append optional variables
        if self.X is not None:
            out += (self.X[idx],)
        if self.dens_ratio is not None:
            out += (self.dens_ratio[idx],)

        return out


class IVDataset(data.Dataset):
    """
    PyTorch Dataset for Instrumental Variable (IV) causal learning.

    Variables are handled in the sequence: A, Y, Z, X (optional), SF (optional).

    Parameters
    ----------
    data_tuple : tuple
        Tuple containing data arrays (A, Y, Z, [X], [SF]).
        Minimum is 3 elements (A, Y, Z). Maximum is 5 (A, Y, Z, X, SF).

    target_type : str, default="Y"
        Which target to use: "Y", "SF", or None.

    transformers : list, optional
        List of transformers for (A, Y, Z, X, SF).
        Uses indices: A[0], Y[1], Z[2], X[3], SF[4].

    device : str, default="cpu"
        Torch device on which to store the tensors.
    """

    def __init__(
        self,
        data_tuple: Union[Tuple, List],
        target_type: str = "Y",
        transformers: Optional[List] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.target_type = target_type
        self.device = device

        # Default transformers = identity for A, Y, Z, X, SF (5 variables)
        if transformers is None:
            transformers = [TorchIdentityTransformer() for _ in range(5)]

        if len(transformers) < 3:
             raise ValueError("The 'transformers' list must have at least 3 elements for (A, Y, Z).")
        self.transformers = transformers

        # --- Unpack data tuple based on length ---
        A, Y, Z, X, SF = None, None, None, None, None

        if len(data_tuple) == 3:
            A, Y, Z = data_tuple
        elif len(data_tuple) == 4:
            A, Y, Z, fourth = data_tuple
            # fourth could be X or SF. If target is SF, assume it's SF.
            if target_type == "SF":
                SF = fourth
            else:
                X = fourth
        elif len(data_tuple) == 5:
            A, Y, Z, X, SF = data_tuple
        else:
            raise ValueError("data_tuple must have 3, 4, or 5 elements: (A, Y, Z, [X], [SF]).")

        # Consistency check for target_type
        if self.target_type == "SF" and SF is None:
            raise ValueError("Target 'SF' requested, but SF data was not provided in data_tuple.")

        self.size = A.shape[0]

        # --- Apply transformations and convert to torch tensors ---

        # 0: A
        A = self.transformers[0].fit_transform(A)
        self.A = torch.as_tensor(A, dtype=torch.float32, device=self.device)

        # 1: Y
        if Y is not None:
            Y = self.transformers[1].fit_transform(Y)
        self.Y = None if Y is None else torch.as_tensor(Y, dtype=torch.float32, device=self.device)

        # 2: Z
        Z = self.transformers[2].fit_transform(Z)
        self.Z = torch.as_tensor(Z, dtype=torch.float32, device=self.device)

        # 3: X (optional)
        if X is not None:
            X = self.transformers[3].fit_transform(X)
        self.X = None if X is None else torch.as_tensor(X, dtype=torch.float32, device=self.device)

        # 4: SF (optional, required if target_type="SF")
        if SF is not None:
            SF = self.transformers[4].fit_transform(SF)
        self.SF = None if SF is None else torch.as_tensor(SF, dtype=torch.float32, device=self.device)

        # Move transformers to device
        for i in range(len(self.transformers)):
            self.transformers[i] = self.transformers[i].to(device)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        """
        Returns the data tuple for a given index.
        Output structure: (A[idx], target[idx], Z[idx]) [+ X[idx] if X is present]
        OR (A[idx], Z[idx]) [+ X[idx] if X is present]
        """
        # --- Determine the target ---
        if self.target_type == "Y":
            target = self.Y
        elif self.target_type == "SF":
            target = self.SF
        else: # self.target_type is None
            target = None

        # --- Construct the output tuple ---
        if target is not None:
            # Base returns: (Treatment A, Target, Instrument Z)
            out = (self.A[idx], target[idx], self.Z[idx])
        else:
            # Base returns: (Treatment A, Instrument Z)
            out = (self.A[idx], self.Z[idx])

        # Add the confounder X if it exists
        if self.X is not None:
            out = out + (self.X[idx],)

        return out