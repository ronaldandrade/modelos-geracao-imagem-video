# Modelos de Geração de Imagem e Vídeo

Estudo técnico, pesquisa comparativa e experimentos em código sobre como a inteligência artificial gera imagens e vídeos, do ruído estocástico ao Flow Matching e Transformers espaço-temporais.

## Como rodar

### Requisitos

- Python 3.10+
- PyTorch (a versão CPU basta; os experimentos rodam em menos de um minuto)

```sh
git clone <url-do-repositorio> modelos-geracao-imagem-video
cd modelos-geracao-imagem-video
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 src/toy_diffusion_vs_flow.py
python3 src/spacetime_patching.py
```

## Estrutura

- `docs/pesquisa-tecnica.md`: levantamento técnico com a matemática de difusão, Flow Matching, 3D VAEs e tabela comparativa dos modelos (Flux, SD3.5, Nano Banana, Wan2.1, HunyuanVideo, CogVideoX, Sora, Kling, Seedance, Veo).
- `src/toy_diffusion_vs_flow.py`: compara DDPM (com amostrador DDIM), Rectified Flow e Rectified Flow com reflow em 2D, medindo retidão de trajetória e qualidade por número de passos.
- `src/spacetime_patching.py`: contagem de tokens e tamanho da matriz de atenção para vídeo, com VAE 3D causal e patches espaço-temporais, e verificação da camada de patchify em PyTorch.
- `artigo/como_ia_cria_imagens_e_videos.mdx`: artigo completo formatado para o blog pessoal (Astro / Rams.Log).

## Como funciona

### `src/toy_diffusion_vs_flow.py`
Cria uma distribuição alvo sintética em anel com 8 gaussianas e treina, com o mesmo MLP, três modelos: um DDPM com predição de ruído (amostrado com DDIM determinístico), um Rectified Flow que prevê a velocidade $x_1 - x_0$ e um Rectified Flow retreinado com uma rodada de *reflow* sobre pares (ruído, amostra) gerados pelo primeiro. Mede dois números:

- **Retidão:** comprimento do caminho dividido pela distância entre início e fim, numa trajetória de 200 passos (reta = 1,0).
- **Precisão:** fração de amostras a menos de 0,3 de algum centro, com 1 a 50 passos.

O resultado mostra que só os caminhos condicionais do Flow Matching são retos. O campo aprendido é curvo (retidão ~1,7, parecida com a do DDIM), mas ainda chega a boa qualidade com menos passos que o DDIM, e o reflow endireita a trajetória (~1,0) a ponto de gerar bem em 1 ou 2 passos.

### `src/spacetime_patching.py`
Calcula tokens e o tamanho da matriz de atenção para um vídeo de 121 quadros (~5 s a 24 fps, no formato $4n+1$ exigido por VAEs causais) em 720p. Compara quatro casos:

- patches de pixels 1×16×16;
- VAE 2D com patch 1×2×2;
- VAE 3D causal 4×8×8 com patch 1×2×2 (Wan2.1 / HunyuanVideo);
- o mesmo latente com patch 2×2×2.

Em seguida, instancia uma `Conv3d` real que fatia o latente em tokens e valida as dimensões de saída.

## Limitações

- Os experimentos de Flow Matching usam uma distribuição bidimensional sintética para fins didáticos; os números não se transferem diretamente para imagens em alta resolução.
- Não inclui download de checkpoints reais (Wan2.1, HunyuanVideo etc.); o foco é a matemática e a arquitetura que cabem num computador pessoal.
- Detalhes de modelos fechados (Midjourney, Veo, Kling, Sora, Seedance 2.x, Nano Banana) estão marcados como não divulgados quando o fabricante não publicou a arquitetura.
