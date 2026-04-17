import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from refactor_project.rl.actions import OpType


class CQV_End2End(nn.Module):
    """Classification model that uses a quantum circuit as part of its architecture.
    The model encodes classical input data into quantum states using a parameterized encoding scheme,
    processes the quantum states through a quantum circuit defined by an architecture matrix,
    and then measures the expectation values of the qubits to produce features
    that are fed into a final linear layer (head) to produce a logit for binary classification.
    """

    def __init__(
        self,
        arch_mat: torch.Tensor,
        n_qubits: int = 4,
        enc_lambda: float = np.pi,
        diff_method: str = "adjoint",
        input_dim: int = 49,
        enc_affine_mode: str = "per_feature",
        use_batched_qnode: bool = True,
        enc_alpha_init: float = 1.0,
        enc_beta_init: float = 0.0,
        enc_beta_max: float = 1.0,
        vqc_theta_init_std: float = 0.1,
    ):
        """Initialize the BinaryCQV_End2End model.
        variables:
            arch_mat: A tensor defining the architecture of the quantum circuit.
            n_qubits: The number of qubits in the quantum circuit.
            enc_lambda: A scaling factor for the encoding of classical data into quantum states.
            diff_method: The method used for computing gradients (e.g., "adjoint").
            input_dim: The dimensionality of the input data.
            enc_affine_mode: The mode for affine transformation in encoding (e.g., "per_feature").
            use_batched_qnode: Whether to use a batched QNode for efficient computation.
            enc_alpha_init: Initial value for the alpha parameter in encoding.
            enc_beta_init: Initial value for the beta parameter in encoding.
            enc_beta_max: Maximum value for the beta parameter in encoding.
            vqc_theta_init_std: Standard deviation for initializing VQC parameters.
        return: None
        """
        super().__init__()
        self.n_qubits = int(n_qubits)
        self.register_buffer("arch_mat", arch_mat.to(torch.int64))
        self.enc_lambda = float(enc_lambda)
        self.input_dim = int(input_dim)
        self.use_batched_qnode = bool(use_batched_qnode)

        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.logit_scale_min = 0.5
        self.logit_scale_max = 80.0

        # --- logit clamp control (default ON for safety) ---
        # This is a safety mechanism to prevent the logits
        # from growing too large, which can lead to numerical
        # instability during training.
        self.clamp_logits_enabled = True
        self.logit_clamp_value = 30.0

        # Debug hooks (filled during forward; env can log them)
        # These are useful for monitoring the behavior of the model during training,
        self._dbg_raw_logits_std = float("nan")
        self._dbg_scaled_logits_std = float("nan")
        self._dbg_logit_scale_value = float("nan")

        # ---- trainable affine encoding params (alpha/beta) ----
        # These parameters control the affine transformation
        # applied to the input data before encoding it into quantum states.
        self.enc_affine_mode = str(enc_affine_mode).lower().strip()
        self.enc_beta_max = float(enc_beta_max)

        # Logit scaling control: scale_eval_only
        # This flag determines whether the logit scaling factor should be applied
        # only during evaluation (inference) or also during training.
        self.logit_scale_eval_only = False

        # stable init helpers
        a0 = float(max(enc_alpha_init, 1e-6))
        # inverse-softplus approx: softplus(z)=a0  => z = log(exp(a0)-1)
        # This is used to initialize the alpha parameter
        #  in a way that is stable and avoids issues with
        # very small values.
        alpha_raw0 = float(np.log(np.exp(a0) - 1.0))
        b0 = float(enc_beta_init)
        denom = max(self.enc_beta_max, 1e-6)
        b0n = float(np.clip(b0 / denom, -0.999, 0.999))
        beta_raw0 = float(np.arctanh(b0n))

        if self.enc_affine_mode == "per_feature_qubit":
            # In this mode, we have separate alpha and beta parameters
            # for each feature and each qubit.
            self.enc_alpha_raw = nn.Parameter(
                torch.full((self.n_qubits, self.input_dim), alpha_raw0, dtype=torch.float32)
            )
            self.enc_beta_raw = nn.Parameter(
                torch.full((self.n_qubits, self.input_dim), beta_raw0, dtype=torch.float32)
            )
        elif self.enc_affine_mode == "per_feature":
            # In this mode, we have separate alpha and beta
            # parameters for each feature,
            self.enc_alpha_raw = nn.Parameter(
                torch.full((self.input_dim,), alpha_raw0, dtype=torch.float32)
            )
            self.enc_beta_raw = nn.Parameter(
                torch.full((self.input_dim,), beta_raw0, dtype=torch.float32)
            )
        elif self.enc_affine_mode == "per_qubit":
            # In this mode, we have separate alpha and beta
            # parameters for each qubit.
            self.enc_alpha_raw = nn.Parameter(
                torch.full((self.n_qubits,), alpha_raw0, dtype=torch.float32)
            )
            self.enc_beta_raw = nn.Parameter(
                torch.full((self.n_qubits,), beta_raw0, dtype=torch.float32)
            )
        else:  # "global" (fallback)
            # In this mode, we have a single alpha and beta parameter
            #  that is shared across all features and qubits.
            self.enc_affine_mode = "global"
            self.enc_alpha_raw = nn.Parameter(torch.tensor(alpha_raw0, dtype=torch.float32))
            self.enc_beta_raw = nn.Parameter(torch.tensor(beta_raw0, dtype=torch.float32))
        # create a stable mapping from layer index -> theta index

        # This is used to efficiently look up the parameters
        # for the rotation gates in the quantum circuit based
        # on the architecture matrix.
        self._rot_param_index = {}
        slots = []
        L = int(self.arch_mat.shape[1])
        for layer_idx in range(L):
            op = int(self.arch_mat[2, layer_idx].item())
            t = int(self.arch_mat[1, layer_idx].item())
            ax = int(self.arch_mat[3, layer_idx].item())
            if op == OpType.ROT.value and (t > 0) and (ax > 0):
                idx = len(slots)
                self._rot_param_index[int(layer_idx)] = int(idx)
                slots.append((layer_idx, t - 1, ax))

        # Initialize the theta parameters for the rotation gates in the quantum circuit.
        self.theta = nn.Parameter(torch.zeros(max(len(slots), 1)))
        with torch.no_grad():
            nn.init.normal_(self.theta, mean=0.0, std=vqc_theta_init_std)
            self.theta.data.clamp_(-vqc_theta_init_std * 3, vqc_theta_init_std * 3)

        # device selection
        # The choice of device for running the quantum circuit is crucial for performance.
        if str(diff_method).lower() == "backprop":
            dev = qml.device("default.qubit", wires=self.n_qubits, shots=None)
            self._dev_name = "default.qubit"
        else:
            try:
                dev = qml.device("lightning.gpu", wires=self.n_qubits, shots=None)
                self._dev_name = "lightning.gpu"
            except Exception:
                dev = qml.device("lightning.qubit", wires=self.n_qubits, shots=None)
                self._dev_name = "lightning.qubit"

        def circuit(xi, theta_vec, enc_alpha_raw, enc_beta_raw):
            """Defines the quantum circuit used in the model.
            variables:
                xi: The input data for a single sample 1D.
                theta_vec: The parameters for the rotation gates in the quantum circuit.
                enc_alpha_raw: The raw alpha parameters for affine encoding.
                enc_beta_raw: The raw beta parameters for affine encoding.
            return: A list of expectation values for each qubit, which are used as features for the final classification layer.
            """
            try:
                xi = xi.reshape(-1)
            except Exception:
                xi = qml.math.reshape(xi, (-1,))
            # apply the affine encoding transformation to the input data
            Lloc = self.arch_mat.shape[1]
            for layer_idx in range(Lloc):
                # Extract the operation type, target qubit,
                # and axis from the architecture matrix f
                # or the current layer.
                op = int(self.arch_mat[2, layer_idx].item())
                c = int(self.arch_mat[0, layer_idx].item())
                t = int(self.arch_mat[1, layer_idx].item())
                ax = int(self.arch_mat[3, layer_idx].item())
                f1 = int(self.arch_mat[4, layer_idx].item())

                # Apply the corresponding quantum gate based on the operation type.
                if op == OpType.ENC.value:
                    if t > 0 and ax > 0 and f1 > 0:
                        tgt = t - 1
                        if tgt < 0 or tgt >= self.n_qubits:
                            continue
                        feat_idx = f1 - 1
                        if feat_idx < 0 or feat_idx >= int(xi.shape[0]):
                            continue
                        # --------------------------
                        # Parameterize the encoding transformation using alpha and beta parameters.
                        # This allows the model to learn how to best encode the classical data into quantum states.
                        # angle = enc_lambda * (alpha[...] * x + beta[...])
                        # alpha>0 (softplus), beta bounded (tanh)
                        # Indexing decided by enc_affine_mode.
                        # ---------------------------

                        # Extract the feature value for the current feature index.
                        xval = xi[feat_idx]

                        # Compute the alpha and beta parameters
                        # for the affine transformation based on the specified mode.
                        if self.enc_affine_mode == "per_feature_qubit":
                            a_raw = enc_alpha_raw[tgt, feat_idx]
                            b_raw = enc_beta_raw[tgt, feat_idx]
                        elif self.enc_affine_mode == "per_feature":
                            a_raw = enc_alpha_raw[feat_idx]
                            b_raw = enc_beta_raw[feat_idx]
                        elif self.enc_affine_mode == "per_qubit":
                            a_raw = enc_alpha_raw[tgt]
                            b_raw = enc_beta_raw[tgt]
                        else:  # "global"
                            a_raw = enc_alpha_raw
                            b_raw = enc_beta_raw

                        alpha = F.softplus(a_raw) + 1e-6
                        beta = torch.tanh(b_raw) * self.enc_beta_max

                        angle = self.enc_lambda * (alpha * xval + beta)

                        # Apply the appropriate rotation gate based on the specified axis.
                        if ax == 1:
                            qml.RX(angle, wires=tgt)
                        elif ax == 2:
                            qml.RY(angle, wires=tgt)
                        elif ax == 3:
                            qml.RZ(angle, wires=tgt)

                elif op == OpType.ROT.value:
                    if t > 0 and ax > 0:
                        tgt = t - 1
                        idx = self._rot_param_index.get(int(layer_idx), None)
                        if idx is None:
                            continue
                        if tgt < 0 or tgt >= self.n_qubits:
                            continue
                        ang = theta_vec[int(idx)]
                        if ax == 1:
                            qml.RX(ang, wires=tgt)
                        elif ax == 2:
                            qml.RY(ang, wires=tgt)
                        elif ax == 3:
                            qml.RZ(ang, wires=tgt)

                elif op == OpType.CNOT.value:
                    if c > 0 and t > 0 and c != t:
                        c0 = c - 1
                        t0 = t - 1
                        if (0 <= c0 < self.n_qubits) and (0 <= t0 < self.n_qubits) and (c0 != t0):
                            qml.CNOT(wires=[c0, t0])

            # Measure the expectation value of Z on the first qubit as the output of the circuit.
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        # Initialize the QNode for the quantum circuit, which allows for efficient execution and gradient computation.
        self._qnode = qml.QNode(
            circuit,
            dev,
            interface="torch",
            diff_method=diff_method,
            cache=True,
            max_diff=1,
        )

        self._qnode_batched = None

        # If the use_batched_qnode flag is set,
        # attempt to create a batched version of the QNode for more efficient
        # computation when processing multiple samples at once.
        if self.use_batched_qnode:
            try:
                qfunc_b = qml.batch_input(circuit, argnum=0)
                self._qnode_batched = qml.QNode(
                    qfunc_b,
                    dev,
                    interface="torch",
                    diff_method=None,
                    cache=True,
                )
            except Exception:
                self._qnode_batched = None
                self.use_batched_qnode = False

        # Initialize the final linear layer (head) that maps the output of the quantum circuit to a single logit for binary classification.
        self.head = nn.Linear(self.n_qubits, 1)

        # Initialize the weights of the head layer using a normal distribution
        # with mean 0 and standard deviation 0.02,
        # which is a common practice for stable training.
        with torch.no_grad():
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def measure_depth_and_cnot(self, x_sample=None):
        """Measures the depth and CNOT count of the quantum circuit for a given sample.
        variables:
            x_sample: A sample input used to construct the quantum circuit for measurement.
            return: A tuple containing the depth of the circuit and the count of CNOT gates.
        """
        if x_sample is None:
            x_sample = torch.zeros(
                (1, self.input_dim), dtype=torch.float32, device=self.theta.device
            )
        xi = x_sample[0]
        tape = self._qnode.construct(
            [
                xi.detach().cpu(),
                self.theta.detach().cpu(),
                self.enc_alpha_raw.detach().cpu(),
                self.enc_beta_raw.detach().cpu(),
            ],
            {},
        )

        qubit_timeline = {}
        depth = 0
        cnot_count = 0
        for op in tape.operations:
            wires = op.wires.tolist()
            if op.name.upper() in ("CNOT", "CX"):
                cnot_count += 1
            layer = max([qubit_timeline.get(w, 0) for w in wires], default=0)
            for w in wires:
                qubit_timeline[w] = layer + 1
            depth = max(depth, layer + 1)
        return depth, cnot_count

    @torch.no_grad()
    def measure_depth_cnot_mean(self, X: torch.Tensor, n_samples: int = 16, seed: int = 0):
        """
        Publication-grade cost measurement:
        measure depth/CNOT from actual tape on random inputs and average.
        variables:
            X: A tensor containing the input data for which to measure the depth and CNOT count.
            n_samples: The number of random samples to use for measurement.
            seed: The random seed for reproducibility.
        return: A tuple containing the average depth and average CNOT count across the sampled inputs.
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Ensure n_samples is a positive integer and does not exceed the number of samples in X.
        n = int(min(int(n_samples), int(X.shape[0])))
        if n <= 0:
            n = int(X.shape[0])

        # Set up a random generator with the specified seed for reproducibility.
        g = torch.Generator(device=X.device)
        g.manual_seed(int(seed))

        # Randomly permute the indices of the input samples and select the first n indices for measurement.
        idx = torch.randperm(X.shape[0], generator=g, device=X.device)[:n]
        depths = []
        cnots = []

        # For each selected sample, run the quantum circuit to ensure it is compiled, then measure and record the depth and CNOT count.
        for i in idx.tolist():
            x1 = X[i : i + 1]
            _ = self(x1)  # ensure qnode compiled
            d, c = self.measure_depth_and_cnot(x1)
            depths.append(float(d))
            cnots.append(float(c))
        if len(depths) == 0:
            return 0.0, 0.0

        # Return the average depth and average CNOT count across the sampled inputs.
        return float(np.mean(depths)), float(np.mean(cnots))

    def _ensure_BD(self, x: torch.Tensor) -> torch.Tensor:
        """Ensures that the input tensor x has the correct shape (B, D) for processing by the model.
        variables:
            x: The input tensor to be reshaped if necessary.
        return: The reshaped tensor with shape (B, D) if it was not already in that shape.
        """
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            if x.numel() == self.input_dim:
                x = x.unsqueeze(0)  # (1,D)
            else:
                x = x.view(-1, 1)  # (B,1)
        elif x.dim() > 2:
            x = x.view(x.shape[0], -1)

        if x.dim() != 2:
            raise RuntimeError(f"[forward] expected 2D (B,D); got {tuple(x.shape)}")

        if x.shape[1] != self.input_dim:
            raise RuntimeError(
                f"[forward] Bad input_dim: got x.shape={tuple(x.shape)} but input_dim={self.input_dim}"
            )

        return x

    def _ev_to_row(self, ev, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Converts the expectation values (ev) obtained from the quantum circuit
        into a row tensor that can be processed by the final linear layer of the model.

        variables:
            ev: The expectation values obtained from the quantum circuit, which can be in various formats (e.g., list, tuple, or tensor).
            device: The device on which the resulting tensor    should be allocated (e.g., CPU or GPU).
            dtype: The data type for the resulting tensor (e.g., torch.float32).

        return: A tensor containing the expectation values reshaped into a row format suitable for input
        to the final linear layer of the model. The tensor will have a shape of (n_qubits,)
        and will be on the specified device with the specified data type.
        """
        if isinstance(ev, (list, tuple)):
            ev = [t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev]
            row = torch.stack(
                [t.to(device=device, dtype=dtype).reshape(()) for t in ev], dim=0
            )  # (nq,)
        else:
            row = ev if torch.is_tensor(ev) else torch.as_tensor(ev, device=device)
            row = row.to(device=device, dtype=dtype).reshape(-1)

        # Ensure that the resulting row tensor has the correct number of elements corresponding to the number of qubits.
        if row.numel() != self.n_qubits:
            # If the row tensor has an extra dimension of size 1, reshape it to remove that dimension.
            if row.dim() == 2 and (
                row.shape == (self.n_qubits, 1) or row.shape == (1, self.n_qubits)
            ):
                row = row.reshape(-1)
            # If the row tensor still does not have the correct number of elements, raise an error indicating the mismatch.
            if row.numel() != self.n_qubits:
                raise RuntimeError(
                    f"Bad expvals row: got shape={tuple(row.shape)} numel={row.numel()} expected n_qubits={self.n_qubits}"
                )

        return row

    def forward(self, x_flat):
        """Defines the forward pass of the BinaryCQV_End2End model,
        which processes the input data through the quantum circuit
        and produces a logit for binary classification.

        variables:
            x_flat: The input data for the model, which can be in various shapes and will
                be reshaped to ensure it has the correct format for processing.

        return: A tensor containing the logit for binary classification,
        which is the output of the model after processing the input through the quantum circuit
        and the final linear layer.
        """
        device = self.theta.device
        dtype = self.head.weight.dtype

        # Reshape the input data to ensure it has the correct shape (B, D) for processing by the model.
        x = x_flat if torch.is_tensor(x_flat) else torch.as_tensor(x_flat)
        x = x.to(device=device, dtype=torch.float32)
        x = self._ensure_BD(x)

        # Compute the alpha and beta parameters for the affine encoding transformation based on the specified mode.
        use_fast_batched = (
            (not torch.is_grad_enabled())
            and self.use_batched_qnode
            and (self._qnode_batched is not None)
        )

        # Depending on whether the fast batched QNode is available and can be used,
        # compute the expectation values from the quantum circuit for the input data.
        # If the fast batched QNode is available and can be used,
        # it will be used to compute the expectation values
        # for all samples in the batch at once, which can significantly improve performance.
        if use_fast_batched:
            try:
                # If the fast batched QNode is available, we can directly pass the entire batch of input data to it.
                x_np = x.detach().cpu().numpy()
                # The expectation values are computed for the entire batch of input data at once,
                # which can be more efficient than processing each sample individually.
                ev = self._qnode_batched(x_np, self.theta, self.enc_alpha_raw, self.enc_beta_raw)

                if isinstance(ev, (list, tuple)):
                    # If the expectation values are returned as a list or tuple,
                    # we need to convert them into a tensor format that
                    # can be processed by the final linear layer.
                    cols = [
                        t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev
                    ]
                    expvals = torch.stack(
                        [c.to(device=device, dtype=dtype).reshape(-1) for c in cols], dim=1
                    )
                else:
                    # If the expectation values are returned as a single tensor,
                    # we need to ensure that it is in the correct format and on the correct device for processing.
                    expvals = ev.to(device=device, dtype=dtype)
                    if expvals.dim() == 1:
                        expvals = expvals.view(1, -1)

            except ValueError:
                # PennyLane may raise a ValueError if the batched QNode fails (e.g., due to unsupported operations or shapes).
                # In this case, we catch the exception and fall back to processing each sample individually.
                B = int(x.shape[0])
                rows = []
                for i in range(B):
                    ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                    rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
                expvals = torch.stack(rows, dim=0)
        else:
            # If the fast batched QNode is not available or cannot be used, we need to process each sample in the batch individually.
            B = int(x.shape[0])
            rows = []
            for i in range(B):
                ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
            expvals = torch.stack(rows, dim=0)

        raw_logits = self.head(expvals)

        # -----------------------------
        # Logit scaling control: scale_eval_only
        # This mechanism allows us to apply the logit scaling factor only during evaluation,
        # while keeping the raw logits during training. This can help stabilize training by preventing the model
        # from compensating for the scaling factor by adjusting the weights of the final linear layer (head).
        # -----------------------------
        scale_eval_only = bool(self.logit_scale_eval_only)

        if self.training and scale_eval_only:
            # DURING TRAINING: do not apply logit scaling
            # (use raw logits) to allow the model to learn without
            # the influence of the scaling factor.
            scale_t = raw_logits.new_tensor(1.0)
        else:
            # DURING EVALUATION (or if scale_eval_only is False): apply logit scaling
            scale_t = torch.clamp(
                self.logit_scale.to(device=device, dtype=raw_logits.dtype),
                min=self.logit_scale_min,
                max=self.logit_scale_max,
            )

        logits = raw_logits * scale_t

        # DEBUG: capture the standard deviation of the raw logits,
        # the scaled logits, and the value of the logit scale for monitoring during training and evaluation.
        # This can help identify issues with exploding or vanishing
        # logits and ensure that the scaling factor is being applied correctly.
        try:
            with torch.no_grad():
                self._dbg_raw_logits_std = float(raw_logits.detach().std().cpu().item())
                self._dbg_scaled_logits_std = float(logits.detach().std().cpu().item())
                self._dbg_logit_scale_value = float(scale_t.detach().cpu().item())
                self._dbg_logit_scale_eval_only = float(scale_eval_only)
                self._dbg_training_flag = float(self.training)
        except Exception:
            pass

        # To prevent numerical instability during training,
        # we apply a safety mechanism to ensure that the logits
        # do not contain NaN or infinite values.
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

        if bool(self.clamp_logits_enabled):
            # The clamp_logits_enabled flag is a safety mechanism
            # to prevent the logits from growing too large,
            # which can lead to numerical instability during training.
            clamp = float(self.logit_clamp_value)
            logits = torch.clamp(logits, min=-clamp, max=clamp)

        return logits

    @torch.no_grad()
    def forward_raw_logits(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Returns logits BEFORE applying logit_scale and clamp.
        Useful for deterministic scale calibration, ablations, and debugging.

        variables:
            x_flat: The input data for the model, which can be in various shapes and will
                be reshaped to ensure it has the correct format for processing.

        return: A tensor containing the raw logits for binary classification,
        which are the outputs of the final linear layer of the model before any scaling or clamping is applied.
        """

        # Reshape the input data to ensure it has the correct shape (B, D) for processing by the model, using theta
        # device and head weight dtype for consistency.
        device = self.theta.device
        dtype = self.head.weight.dtype

        x = x_flat if torch.is_tensor(x_flat) else torch.as_tensor(x_flat)
        x = x.to(device=device, dtype=torch.float32)
        x = self._ensure_BD(x)

        # Compute the alpha and beta parameters for the affine encoding transformation based on the specified mode.
        use_fast_batched = (
            (not torch.is_grad_enabled())
            and self.use_batched_qnode
            and (self._qnode_batched is not None)
        )

        if use_fast_batched:
            try:
                """If the fast batched QNode is available, we can directly pass the entire batch of input data to it.
                The expectation values are computed for the entire batch of input data at once, 
                which can be more efficient than processing each sample individually.
                """
                x_np = x.detach().cpu().numpy()
                ev = self._qnode_batched(x_np, self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                if isinstance(ev, (list, tuple)):
                    cols = [
                        t if torch.is_tensor(t) else torch.as_tensor(t, device=device) for t in ev
                    ]
                    expvals = torch.stack(
                        [c.to(device=device, dtype=dtype).reshape(-1) for c in cols], dim=1
                    )
                else:
                    expvals = ev.to(device=device, dtype=dtype)
                    if expvals.dim() == 1:
                        expvals = expvals.view(1, -1)

            except Exception:
                """PennyLane may raise an exception if the batched QNode fails (e.g., due to unsupported operations or shapes).
                In this case, we catch the exception and fall back to processing each sample individually.
                """
                B = int(x.shape[0])
                rows = []
                for i in range(B):
                    ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                    rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
                expvals = torch.stack(rows, dim=0)

        else:
            """If the fast batched QNode is not available or cannot be used, we need to process each sample in the batch individually."""
            B = int(x.shape[0])
            rows = []
            for i in range(B):
                ev_i = self._qnode(x[i], self.theta, self.enc_alpha_raw, self.enc_beta_raw)
                rows.append(self._ev_to_row(ev_i, device=device, dtype=dtype))
            expvals = torch.stack(rows, dim=0)

        # Compute the raw logits by passing the expectation values through the final linear layer (head) of the model.
        raw_logits = self.head(expvals)
        return torch.nan_to_num(raw_logits, nan=0.0, posinf=1e6, neginf=-1e6)

    @torch.no_grad()
    def set_logit_scale_calibrated(
        self,
        raw_logits: torch.Tensor,
        method: str = "p95",
        target: float = 8.0,
        eps: float = 1e-6,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ) -> float:
        """
        Deterministic (non-learned) scaling to reduce seed-to-seed temperature drift.
        raw_logits must be BEFORE applying self.logit_scale.
        Using the specified method, computes a scaling factor to calibrate the raw logits such that a certain statistic
        (95th percentile or std) of the scaled logits matches the target value.
        variables:
            raw_logits: A tensor containing the raw logits for binary classification,
                which are the outputs of the final linear layer of the
                model before any scaling or clamping is applied.

            method: The method used for calibration, which can be either "p95" to match
                the 95th percentile of the scaled logits to the target value, or "std" to match the standard
                deviation of the scaled logits to the target value.
                target: The target value that the specified statistic (95th percentile or standard deviation)
                of the scaled logits should match after applying the computed scaling factor.

            eps: A small value added to the denominator for numerical stability when computing the scaling factor.

            clamp_min: An optional minimum value to clamp the computed scaling factor to,
                which can help prevent excessively small scaling factors
                that could lead to vanishing gradients.
            clamp_max: An optional maximum value to clamp the
                computed scaling factor to, which can help prevent
                excessively large scaling factors that could
                lead to exploding gradients.
        return: The computed scaling factor that has been
            set to the logit_scale parameter of the model, which is used to scale the logits
            during evaluation (or during training if logit_scale_eval_only is False).
        """
        x = raw_logits.detach().cpu().view(-1).float()
        # If the input tensor is empty, we cannot compute the desired statistic (95th percentile or standard deviation) for calibration.
        if x.numel() == 0:
            return float(self.logit_scale.detach().cpu().item())

        method = str(method).lower().strip()
        if method == "std":
            denom = float(x.std().cpu().item())
        else:
            denom = float(torch.quantile(torch.abs(x), 0.95).cpu().item())
        # The scaling factor is computed as the ratio of the target value to the computed statistic (denominator),
        # which is either the standard deviation or the 95th percentile of the raw logits.
        # A small value (eps) is added to the denominator for numerical stability to prevent division by zero.

        s = float(target) / float(max(denom, eps))
        lo = float(clamp_min) if clamp_min is not None else float(self.logit_scale_min)
        hi = float(clamp_max) if clamp_max is not None else float(self.logit_scale_max)
        s = float(max(lo, min(hi, s)))
        self.logit_scale.fill_(float(s))
        return float(s)

    def set_logit_scale_trainable(self, trainable: bool, value: float | None = None):
        """
        If trainable=False, freezes logit_scale (used in SEARCH).
        If trainable=True, enables learning (used in FINAL).
        Optionally sets a fixed value.
        variables:
            trainable: A boolean flag indicating whether the logit_scale parameter should be trainable (learnable) or not.
                If set to False, the logit_scale will be frozen and will not be updated during training.
                If set to True, the logit_scale will be learnable and can be updated during training.

            value: An optional float value to set the logit_scale to.
                If provided, this value will be assigned to the logit_scale parameter regardless of whether it is trainable or not.

        return: None. This method updates the logit_scale parameter of the model based on the specified trainable
        flag and optional value, but does not return any value.
        """
        if value is not None:
            with torch.no_grad():
                self.logit_scale.fill_(float(value))
        self.logit_scale.requires_grad_(bool(trainable))

    def set_clamp_logits(self, enabled: bool, clamp_value: float | None = None) -> None:
        """Enables or disables logit clamping as a safety mechanism to prevent exploding logits during training,
        and optionally sets the clamp value.
        variables:
            enabled: A boolean flag indicating whether logit clamping should be enabled or disabled.
            clamp_value: An optional float value to set the clamp value to. If None, no clamp value is set.
        return: None. This method updates the clamp_logits_enabled flag and optionally
        sets the logit_clamp_value based on the provided parameters.
        """
        self.clamp_logits_enabled = bool(enabled)
        if clamp_value is not None:
            self.logit_clamp_value = float(clamp_value)
