A small NumPy exercise estimating a 2D position from three noisy distance measurements.
It implements distance residuals, their analytic Jacobian, damped least-squares
steps, and a simple accept/reject rule to adjusts damping.

```sh
uv run --with numpy --with matplotlib exercise.py
```
