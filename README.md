# Modelos de Geração de Imagem e Vídeo

Estudo técnico, pesquisa comparativa e experimentos em código sobre como a inteligência artificial gera imagens e vídeos, do ruído estocástico ao Flow Matching e Transformers espaço-temporais.

## Como rodar

### Requisitos

- Linux com Python 3.10+
- PyTorch instalado (`torch`)

```sh
git clone /home/ronald/projetos/modelos-geracao-imagem-video.git
cd modelos-geracao-imagem-video
/usr/bin/python3 src/toy_diffusion_vs_flow.py
/usr/bin/python3 src/spacetime_patching.py
```

## Estrutura

- `docs/pesquisa-tecnica.md`: levantamento técnico com a matemática de difusão, Flow Matching, 3D VAEs e tabela comparativa dos modelos (Flux, SD3.5, Wan2.1, HunyuanVideo, Sora, Kling).
- `src/toy_diffusion_vs_flow.py`: implementação minimalista em PyTorch de Flow Matching (Rectified Flow) e medição de curvatura de trajetória por passos de Euler.
- `src/spacetime_patching.py`: cálculo de dimensionalidade, projeção 3D de patches latentes e análise da redução quadrática de atenção em vídeo.
- `artigo/como_ia_cria_imagens_e_videos.mdx`: artigo completo formatado para o blog pessoal (Astro / Rams.Log).

## Como funciona

### `src/toy_diffusion_vs_flow.py`
O script cria uma distribuição alvo sintética em formato de anel com 8 centros gaussianos. Em vez de usar schedules de ruído curvos de difusão clássica, treina um perceptron multicamadas pequeno para prever o vetor velocidade linear entre ruído padrão e os pontos de dados reais. Depois, executa a integração numérica de Euler para 5, 10, 20 e 50 passos e calcula a curvatura média da trajetória via similaridade de cosseno entre passos consecutivos, demonstrando por que o Flow Matching preserva a geometria da distribuição mesmo com poucos passos de inferência.

### `src/spacetime_patching.py`
Calcula o número de tokens e o tamanho da matriz de autoatenção para um vídeo de 5 segundos a 24 quadros por segundo em 720p. Compara o método ingênuo de pixels, a difusão latente 2D com extensão temporal e a abordagem com 3D Causal VAE com cubos espaço-temporais de 2x2x2. Em seguida, instancia uma camada convolucional 3D real no PyTorch que fatia um tensor latente de vídeo em tokens prontos para um Transformer, validando as dimensões exatas de saída.

## Limitações

- O script de Flow Matching usa uma distribuição bidimensional sintética para fins didáticos e não treina pesos para geração de imagens fotográficas em alta resolução.
- Não inclui download de checkpoints de 14 bilhões de parâmetros (como Wan2.1 ou HunyuanVideo), focando na matemática e na arquitetura subjacente que cabe em computadores pessoais.
