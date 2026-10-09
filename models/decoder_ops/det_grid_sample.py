"""确定性双线性 grid_sample（A4）。

数学等价于 F.grid_sample(mode='bilinear', align_corners=False, padding_mode='border')。

动机: torch 2.4.1 的 grid_sample CUDA 反向(grid_sampler_2d_backward_cuda)无确定性实现
(atomicAdd 累加), 使 DySample 实验间不可复现。本模块用自定义 autograd.Function 重写反向:
  - 前向: 与 F.grid_sample 完全一致(同样取 4 邻域双线性 + border 边界裁剪);
  - 反向: 用 index_put_(accumulate=True)(实测逐位确定)做散回, 替换 atomicAdd。
数学上与 grid_sample 相同, 仅固定了求和顺序, 故结果一致、过程可复现。
"""
import torch


class _DeterministicGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grid):
        # x: [N, C, Hin, Win]; grid: [N, Hout, Wout, 2] (归一化 [-1,1], 末维 (x, y))
        N, C, Hin, Win = x.shape
        Hout, Wout = grid.shape[1], grid.shape[2]
        # 归一化 -> 像素坐标 (align_corners=False)
        ix = ((grid[..., 0] + 1.0) * Win - 1.0) * 0.5
        iy = ((grid[..., 1] + 1.0) * Hin - 1.0) * 0.5
        ix_c = ix.clamp(0.0, Win - 1.0)
        iy_c = iy.clamp(0.0, Hin - 1.0)
        x0 = torch.floor(ix_c)
        y0 = torch.floor(iy_c)
        dx = ix_c - x0
        dy = iy_c - y0
        x0i = x0.to(torch.long)
        y0i = y0.to(torch.long)
        x1i = (x0i + 1).clamp(0, Win - 1)
        y1i = (y0i + 1).clamp(0, Hin - 1)

        xf = x.reshape(N, C, Hin * Win)
        base_nc = torch.arange(N * C, device=x.device).view(N, C, 1, 1)

        def gather_at(yy, xx):
            pos = (yy * Win + xx).unsqueeze(1)                       # [N,1,Hout,Wout]
            idx = pos.expand(N, C, Hout, Wout).reshape(N, C, Hout * Wout)
            return xf.gather(2, idx).reshape(N, C, Hout, Wout)

        v00 = gather_at(y0i, x0i)
        v01 = gather_at(y0i, x1i)
        v10 = gather_at(y1i, x0i)
        v11 = gather_at(y1i, x1i)

        dx1 = dx.unsqueeze(1)
        dy1 = dy.unsqueeze(1)
        w00 = (1 - dx1) * (1 - dy1)
        w01 = dx1 * (1 - dy1)
        w10 = (1 - dx1) * dy1
        w11 = dx1 * dy1
        out = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11

        ctx.save_for_backward(x, ix, iy, dx, dy, x0i, y0i, x1i, y1i)
        ctx.dims = (Hin, Win, Hout, Wout, C)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, ix, iy, dx, dy, x0i, y0i, x1i, y1i = ctx.saved_tensors
        Hin, Win, Hout, Wout, C = ctx.dims
        N = x.shape[0]
        dev = x.device
        grad_out = grad_out.contiguous()

        dx1 = dx.unsqueeze(1)
        dy1 = dy.unsqueeze(1)
        w00 = (1 - dx1) * (1 - dy1)
        w01 = dx1 * (1 - dy1)
        w10 = (1 - dx1) * dy1
        w11 = dx1 * dy1

        # ---- grad wrt x: 确定性散回 (index_put_ accumulate) ----
        grad_x = torch.zeros(N * C * Hin * Win, dtype=x.dtype, device=dev)
        base_nc = torch.arange(N * C, device=dev).view(N, C, 1, 1)

        def scatter_at(yy, xx, w):
            pos = (yy * Win + xx).unsqueeze(1)
            flat = (base_nc * (Hin * Win) + pos).reshape(-1)
            vals = (grad_out * w).reshape(-1)
            grad_x.index_put_((flat,), vals, accumulate=True)

        scatter_at(y0i, x0i, w00)
        scatter_at(y0i, x1i, w01)
        scatter_at(y1i, x0i, w10)
        scatter_at(y1i, x1i, w11)
        grad_x = grad_x.reshape(N, C, Hin, Win)

        # ---- grad wrt grid ----
        xf = x.reshape(N, C, Hin * Win)

        def val(yy, xx):
            pos = (yy * Win + xx).unsqueeze(1)
            idx = pos.expand(N, C, Hout, Wout).reshape(N, C, Hout * Wout)
            return xf.gather(2, idx).reshape(N, C, Hout, Wout)

        x00 = val(y0i, x0i)
        x01 = val(y0i, x1i)
        x10 = val(y1i, x0i)
        x11 = val(y1i, x1i)

        gix = (1 - dy1) * (x01 - x00) + dy1 * (x11 - x10)   # d out / d ix
        giy = (1 - dx1) * (x10 - x00) + dx1 * (x11 - x01)   # d out / d iy
        grad_ix = (grad_out * gix).sum(dim=1)               # [N,Hout,Wout]
        grad_iy = (grad_out * giy).sum(dim=1)

        # 边界裁剪区间内导数为 0 (d clamp / d coord = 0)
        mask_x = ((ix > 0) & (ix < Win - 1)).to(x.dtype)
        mask_y = ((iy > 0) & (iy < Hin - 1)).to(x.dtype)
        grad_ux = grad_ix * mask_x * (Win * 0.5)
        grad_uy = grad_iy * mask_y * (Hin * 0.5)
        grad_grid = torch.stack([grad_ux, grad_uy], dim=-1)

        return grad_x, grad_grid


def deterministic_grid_sample(x, grid):
    """drop-in 替代 F.grid_sample(x, grid, mode='bilinear', align_corners=False, padding_mode='border')。"""
    return _DeterministicGridSample.apply(x, grid)
