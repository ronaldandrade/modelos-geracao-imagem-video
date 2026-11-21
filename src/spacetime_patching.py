import torch
import torch.nn as nn

class SpacetimePatchEmbedding(nn.Module):
    def __init__(self, in_channels=16, embed_dim=512, patch_size=(2, 2, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        proj = self.proj(x)
        b, d, t, h, w = proj.shape
        tokens = proj.flatten(2).transpose(1, 2)
        return tokens, (t, h, w)

def analyze_video_token_explosion():
    fps = 24
    duration_sec = 5
    total_frames = fps * duration_sec
    height = 720
    width = 1280
    channels = 3
    
    raw_pixels = total_frames * height * width * channels
    raw_bytes_fp16 = raw_pixels * 2
    raw_mb = raw_bytes_fp16 / (1024 ** 2)
    
    p_t, p_h, p_w = 1, 16, 16
    tokens_pixel_naive = total_frames * (height // p_h) * (width // p_w)
    
    vae_t_scale = 1
    vae_s_scale = 8
    c_lat_2d = 4
    t_lat_2d = total_frames // vae_t_scale
    h_lat_2d = height // vae_s_scale
    w_lat_2d = width // vae_s_scale
    tokens_lat_2d = t_lat_2d * (h_lat_2d // 2) * (w_lat_2d // 2)
    
    c_lat_3d = 16
    vae_3d_t_scale = 4
    vae_3d_s_scale = 8
    t_lat_3d = total_frames // vae_3d_t_scale
    h_lat_3d = height // vae_3d_s_scale
    w_lat_3d = width // vae_3d_s_scale
    
    patch_t, patch_h, patch_w = 2, 2, 2
    tokens_3d_dit = (t_lat_3d // patch_t) * (h_lat_3d // patch_h) * (w_lat_3d // patch_w)
    
    print("=================================================================")
    print(" ANALISE DE DIMENSIONALIDADE E ATENCAO ESPACO-TEMPORAL EM VIDEO")
    print("=================================================================")
    print(f"Video de entrada: {duration_sec}s @ {fps}fps ({total_frames} frames) em {width}x{height}")
    print(f"Volume de pixels brutos: {raw_pixels:,} elementos (~{raw_mb:.1f} MB em FP16 por clipe)")
    print("-----------------------------------------------------------------")
    print(f"1. Naive Pixel Transformer (patches {p_t}x{p_h}x{p_w}):")
    print(f"   Tokens de sequencia: {tokens_pixel_naive:,}")
    print(f"   Matriz de Atencao (N^2): {tokens_pixel_naive**2:,} elementos")
    print("-----------------------------------------------------------------")
    print(f"2. Latent 2D VAE (SVD / AnimateDiff style, compressao espacial 8x):")
    print(f"   Tokens latentes (patch 1x2x2): {tokens_lat_2d:,}")
    print(f"   Matriz de Atencao (N^2): {tokens_lat_2d**2:,} elementos")
    print("-----------------------------------------------------------------")
    print(f"3. 3D Causal VAE + DiT (Wan2.1 / Sora / HunyuanVideo style):")
    print(f"   Latente 3D: {t_lat_3d} frames x {h_lat_3d}x{w_lat_3d} canais={c_lat_3d}")
    print(f"   Tokens 3D (patch {patch_t}x{patch_h}x{patch_w}): {tokens_3d_dit:,}")
    print(f"   Matriz de Atencao (N^2): {tokens_3d_dit**2:,} elementos")
    
    reducao = tokens_pixel_naive / tokens_3d_dit
    reducao_atencao = (tokens_pixel_naive**2) / (tokens_3d_dit**2)
    print("-----------------------------------------------------------------")
    print(f"Fator de reducao no comprimento da sequencia: {reducao:.1f}x")
    print(f"Fator de reducao na matriz de atencao: {reducao_atencao:.1f}x")
    print("=================================================================")

def verify_spacetime_embedding():
    batch = 1
    channels = 16
    t_lat, h_lat, w_lat = 30, 90, 160
    
    patcher = SpacetimePatchEmbedding(in_channels=channels, embed_dim=256, patch_size=(2, 2, 2))
    dummy_latent = torch.randn(batch, channels, t_lat, h_lat, w_lat)
    tokens, grid = patcher(dummy_latent)
    
    print("\nVerificacao do modulo PyTorch:")
    print(f"Entrada latente 3D: {list(dummy_latent.shape)}")
    print(f"Grade de patches (T, H, W): {grid}")
    print(f"Tensor de tokens para o Transformer: {list(tokens.shape)} (Batch, Tokens, Embedding)")
    assert tokens.shape == (1, 15 * 45 * 80, 256)
    print("Teste concluido com sucesso!")

if __name__ == "__main__":
    analyze_video_token_explosion()
    verify_spacetime_embedding()
