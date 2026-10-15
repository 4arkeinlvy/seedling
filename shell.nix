# Dev shell for seedling. Enter with:  nix-shell
# Python and PyTorch come from uv (see README), not from Nix.
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    gcc
    gnumake
    gdb
    valgrind
  ];

  # Nix's compiler wrapper drops -march=native by default (for reproducible builds). We
  # want AVX2 for this laptop, so allow it.
  NIX_ENFORCE_NO_NATIVE = 0;

  # PyTorch (installed by uv) loads the NVIDIA driver library libcuda.so.1 by name. On
  # NixOS it lives in /run/opengl-driver/lib, which the loader doesn't search by default.
  # Without this, torch.cuda.is_available() is False. uv's Python also needs to be told
  # where NixOS keeps the TLS certificate bundle.
  shellHook = ''
    export LD_LIBRARY_PATH=/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    export SSL_CERT_FILE=''${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}
  '';
}
