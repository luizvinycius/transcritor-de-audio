"""Transcreve um arquivo de áudio para texto usando faster-whisper.

Uso:
    python transcrever.py caminho/do/audio.mp3
    python transcrever.py audio.m4a --modelo turbo --idioma pt
"""

import argparse
import os
import sys
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

MODELOS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
DISPOSITIVOS = ["auto", "cpu", "cuda"]


def registrar_dlls_nvidia() -> None:
    """No Windows, coloca no PATH as DLLs CUDA instaladas via requirements-gpu.txt."""
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    pastas = [str(p) for base in nvidia.__path__ for p in Path(base).glob("*/bin")]
    os.environ["PATH"] = os.pathsep.join([*pastas, os.environ.get("PATH", "")])


def escolher_dispositivo(pedido: str) -> tuple[str, str]:
    """Retorna (dispositivo, compute_type); 'auto' usa a GPU NVIDIA se houver."""
    tem_gpu = ctranslate2.get_cuda_device_count() > 0
    dispositivo = ("cuda" if tem_gpu else "cpu") if pedido == "auto" else pedido
    return dispositivo, ("float16" if dispositivo == "cuda" else "int8")


def formatar_tempo(segundos: float) -> str:
    minutos, seg = divmod(int(segundos), 60)
    horas, minutos = divmod(minutos, 60)
    return f"{horas:02d}:{minutos:02d}:{seg:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcreve áudio para texto.")
    parser.add_argument("audio", type=Path, help="arquivo de áudio ou vídeo")
    parser.add_argument("--modelo", default="small", choices=MODELOS,
                        help="maior = mais preciso e mais lento (padrão: small)")
    parser.add_argument("--idioma", default="pt",
                        help="código do idioma, ex.: pt, en, es (padrão: pt)")
    parser.add_argument("--dispositivo", default="auto", choices=DISPOSITIVOS,
                        help="auto usa a placa NVIDIA se houver (padrão: auto)")
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"Arquivo não encontrado: {args.audio}", file=sys.stderr)
        return 1

    registrar_dlls_nvidia()
    dispositivo, compute_type = escolher_dispositivo(args.dispositivo)
    print(f"Carregando modelo '{args.modelo}' em {dispositivo} "
          "(o primeiro uso baixa o modelo)...")
    modelo = WhisperModel(args.modelo, device=dispositivo, compute_type=compute_type)

    segmentos, info = modelo.transcribe(str(args.audio), language=args.idioma,
                                        vad_filter=True)
    print(f"Duração do áudio: {formatar_tempo(info.duration)}\n")

    linhas = []
    for seg in segmentos:
        texto = seg.text.strip()
        print(f"[{formatar_tempo(seg.start)}] {texto}")
        linhas.append(texto)

    saida = args.audio.with_suffix(".txt")
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\nTranscrição salva em: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
