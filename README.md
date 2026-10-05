# Transcritor de áudio

Transcreve áudio (ou vídeo) para texto localmente com
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), que usa os modelos
Whisper da OpenAI. Usa a placa NVIDIA automaticamente quando disponível.

## Requisitos

- Python 3.10+
- Para GPU: placa NVIDIA com driver atualizado (CUDA 12)

## Instalação

```powershell
git clone https://github.com/luizvinycius/transcritor-de-audio.git
cd transcritor-de-audio
python -m venv .venv

# Com placa NVIDIA:
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt

# Só processador:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Uso

```powershell
.\.venv\Scripts\python.exe transcrever.py "C:\caminho\do\audio.mp3"
```

O texto aparece na tela com o tempo de cada trecho e é salvo num `.txt` ao
lado do áudio.

| Opção | Valores | Padrão |
|---|---|---|
| `--modelo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo` | `small` |
| `--idioma` | `pt`, `en`, `es`, ... | `pt` |
| `--dispositivo` | `auto`, `cpu`, `cuda` | `auto` |

Modelos maiores são mais precisos e mais lentos; cada um é baixado uma vez no
primeiro uso. Com GPU NVIDIA, `--modelo turbo` ou `--modelo large-v3` dão a
melhor qualidade com boa velocidade. Só no processador, `small` é um bom
equilíbrio.

## Problemas comuns

- **`cublas64_12.dll is not found`**: instale com `requirements-gpu.txt` (não
  só `requirements.txt`) ou atualize o driver da NVIDIA.
- **Para forçar o processador**: `--dispositivo cpu`.
