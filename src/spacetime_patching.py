"""Contagem de tokens e tamanho da matriz de atenção para um clipe de vídeo de ~5 s
em 720p, comparando patches em pixels, VAE 2D e VAE 3D causal (estilo Wan2.1 /
HunyuanVideo), com patch (1, 2, 2), o usado por esses modelos, e com (2, 2, 2), que agrupa também no tempo.
"""

import torch
import torch.nn as nn


class SpacetimePatchEmbedding(nn.Module):
    def __init__(self, in_channels=16, embed_dim=512, patch_size=(1, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        proj = self.proj(x)
        _, _, t, h, w = proj.shape
        tokens = proj.flatten(2).transpose(1, 2)
        return tokens, (t, h, w)


def causal_latent_frames(num_frames, t_scale=4):
    """VAE 3D causal: o 1º quadro é codificado sozinho e o resto em grupos de t_scale.
    Por isso os modelos pedem 4n + 1 quadros (ex.: 81 ou 121)."""
    assert (num_frames - 1) % t_scale == 0, "use 4n + 1 quadros"
    return 1 + (num_frames - 1) // t_scale


def _report(label, n_tokens):
    print(f"   Tokens: {n_tokens:,}")
    print(f"   Matriz de atencao (N^2): {n_tokens**2:,} entradas (FLOPs ~ N^2 * d por camada)")


def analyze_video_token_explosion():
    num_frames = 121  # ~5 s a 24 fps, no formato 4n + 1 exigido pelo VAE causal
    height, width, channels = 720, 1280, 3

    raw = num_frames * height * width * channels
    raw_mib = raw * 2 / 1024**2

    tok_pixel = num_frames * (height // 16) * (width // 16)
    tok_2d = num_frames * (height // 8 // 2) * (width // 8 // 2)

    t_lat = causal_latent_frames(num_frames)
    h_lat, w_lat = height // 8, width // 8
    tok_122 = t_lat * (h_lat // 2) * (w_lat // 2)
    tok_222 = -(-t_lat // 2) * (h_lat // 2) * (w_lat // 2)  # teto: o último grupo temporal é preenchido

    print("=" * 65)
    print(" DIMENSIONALIDADE E ATENCAO ESPACO-TEMPORAL EM VIDEO")
    print("=" * 65)
    print(f"Video: {num_frames} quadros (~5 s @ 24 fps) em {width}x{height}")
    print(f"Pixels brutos: {raw:,} valores (~{raw_mib:.0f} MiB em FP16)")
    print("-" * 65)
    print("1. Transformer em pixels, patches 1x16x16:")
    _report("pixel", tok_pixel)
    print("2. VAE 2D (8x espacial, 4 canais) + patch 1x2x2, quadro a quadro:")
    _report("2d", tok_2d)
    print(f"3. VAE 3D causal 4x8x8 (latente {t_lat}x{h_lat}x{w_lat}x16) + patch 1x2x2 (Wan2.1 / HunyuanVideo):")
    _report("122", tok_122)
    print("4. Mesmo latente + patch 2x2x2 (patch temporal 2, ex.: CogVideoX 1.5):")
    _report("222", tok_222)
    print("-" * 65)
    print(f"Pixels -> VAE 3D + 1x2x2: sequencia {tok_pixel / tok_122:.1f}x menor, atencao {(tok_pixel / tok_122) ** 2:.1f}x menor")
    print(f"Pixels -> VAE 3D + 2x2x2: sequencia {tok_pixel / tok_222:.1f}x menor, atencao {(tok_pixel / tok_222) ** 2:.1f}x menor")
    print("=" * 65)
    return t_lat, h_lat, w_lat


def verify_spacetime_embedding(t_lat, h_lat, w_lat, channels=16):
    latent = torch.randn(1, channels, t_lat, h_lat, w_lat)
    patcher = SpacetimePatchEmbedding(in_channels=channels, embed_dim=256, patch_size=(1, 2, 2))
    tokens, grid = patcher(latent)

    print("\nVerificacao do modulo PyTorch (patch 1x2x2):")
    print(f"Latente 3D: {list(latent.shape)}")
    print(f"Grade de patches (T, H, W): {grid}")
    print(f"Tokens para o Transformer: {list(tokens.shape)} (batch, tokens, embedding)")
    assert tokens.shape == (1, t_lat * (h_lat // 2) * (w_lat // 2), 256)
    print("Teste concluido com sucesso!")


if __name__ == "__main__":
    verify_spacetime_embedding(*analyze_video_token_explosion())
