import torch
import matplotlib.pyplot as plt

# def light_cone_mask(t_q, h_q, w_q, T, H, W, max_radius=4):
#     mask = torch.zeros((T, H, W), dtype=torch.bool)

#     for t in range(t_q):  # only attend to past
#         radius = int((t_q - t) / t_q * max_radius)
#         for dh in range(-radius, radius + 1):
#             for dw in range(-radius, radius + 1):
#                 h = h_q + dh
#                 w = w_q + dw
#                 if 0 <= h < H and 0 <= w < W:
#                     mask[t, h, w] = True
#     return mask


def light_cone_mask(t_q, h_q, w_q, T, H, W):
    mask = torch.zeros((T, H, W), dtype=torch.bool)

    max_possible_radius = max(H, W)  # when reaching entire image
    min_radius = 1  # local at present

    for t in range(t_q + 1):  # only up to and including present
        # Linearly interpolate radius from min_radius (at t_q) to max_radius (at t=0)
        if t_q == 0:
            radius = max_possible_radius
        else:
            radius = int(min_radius + (max_possible_radius - min_radius) * (t_q - t) / t_q)

        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                h = h_q + dh
                w = w_q + dw
                if 0 <= h < H and 0 <= w < W:
                    mask[t, h, w] = True

    return mask





def visualize_light_cone(t_q, h_q, w_q, T, H, W, ):
    mask = light_cone_mask(t_q, h_q, w_q, T, H, W)
    print(mask[0])
    fig, axes = plt.subplots(1, T, figsize=(15, 3))
    for t in range(T):
        axes[t].imshow(mask[t].numpy(), cmap='gray')
        axes[t].set_title(f't = {t}')
        axes[t].axis('off')
    plt.suptitle(f"Light Cone Attention for Query (t={t_q}, h={h_q}, w={w_q})")
    plt.show()


# Try it
visualize_light_cone(t_q=5, h_q=4, w_q=4, T=6, H=16, W=16)
