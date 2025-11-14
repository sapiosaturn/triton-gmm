## Usage

```python
from triton_gmm import tgmm

# Use tgmm exactly like torch._grouped_mm
result = tgmm(activations, weights, offs=offsets)
```

## Auto-patch

To automatically replace `torch._grouped_mm`:

```python
from triton_gmm import tgmm_autopatch

# Now torch._grouped_mm uses the Triton implementation
result = torch._grouped_mm(activations, weights, offs=offsets)
```

