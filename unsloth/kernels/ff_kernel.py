import torch
import triton
import triton.language as tl
from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

def matmul(X, W, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass

    out = torch.matmul(X, W.t(), out = out)
    
    return out.view(batch, seq_len, -1) if reshape else out
pass

class FAST_MLP(torch.autograd.Function):
    """
    ### SwiGLU(X)
    e = X @ G
    f = e * sigmoid(e)
    g = X @ U
    h = f * g
    i = h @ W

    ### Backpropagation chain rule
    See our blog post for more details

    df = sigmoid(e) * (1 - f) + f
    dC/dW = h.T @ dY
    dC/dU = X.T @ (D @ W.T * f)
    dC/dG = X.T @ (D @ W.T * df * g)
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, X : torch.Tensor,
                gateW,
                  upW,
                downW,):
        # dtype = X.dtype

        e = matmul(X, gateW)
        g = matmul(X,   upW)
        h = swiglu_fg_kernel(e, g)
        i = matmul(h, downW)

        ctx.custom_saved_tensors = (
            gateW,
            upW,
            downW
        )
        ctx.save_for_backward(X, e, g)
        return i
    pass

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY : torch.Tensor):
        gateW, upW, downW, = ctx.custom_saved_tensors
        X, e, g = ctx.saved_tensors

        batch, seq_len, hd = X.shape
        dY = dY.view(-1, dY.shape[-1])
        X  = X .view(-1, X .shape[-1])
        e  = e .view(-1, e .shape[-1])
        g  = g .view(-1, g .shape[-1])
        # dtype = X.dtype

        # Transpose weight matrices
        gateW_t = gateW.t()
        upW_t = upW.t()
        downW_t = downW.t()
        del upW
        del gateW
        del downW
        # DW_f   = (D @ W.T * f)
        # DW_dfg = (D @ W.T * df * g)
        DW = matmul(dY, downW_t)
        DW, e, g = swiglu_DWf_DW_dfg_kernel(DW, e, g)
        h, DW_f, DW_dfg = DW, e, g

        dX = torch.matmul(DW_f, upW_t, out = X) + DW_dfg @ gateW_t
        

        # Calculate gradients for weight matrices
        dC_dW = matmul(h.t(), dY)
        dC_dU = matmul(X.t(), matmul(DW_f, upW_t))
        dC_dG = matmul(X.t(), matmul(DW_dfg * e / (1 + tl.exp(-e)), gateW_t))

        # Reshape the gradients to the original shape
        dC_dW = dC_dW.view(batch, seq_len, -1)
        dC_dU = dC_dU.view(batch, seq_len, -1)
        dC_dG = dC_dG.view(batch, seq_len, -1)

        # Return the gradients
        return dX.view(batch, seq_len, hd), dC_dW, dC_dG, dC_dU

def get_parameters(proj):
    # For DPO or disabled adapters
    base_layer = (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight
    return W

def apply_mlp(self, X):
    gateW = get_parameters(self.gate_proj)
    upW = get_parameters(self.  up_proj)
    downW = get_parameters(self.down_proj)
    out = FAST_MLP.apply(X,
                         gateW,
                         upW,
                         downW)
    return out
pass