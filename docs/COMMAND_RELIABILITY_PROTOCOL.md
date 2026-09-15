# Command Reliability Notes

## 2026-09-15: Cloud unit-test import path

Invalid form:

```bash
/sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python \
  -m unittest discover -s tests -v
```

This could not import the package under `src/dehaze` and failed with
`ModuleNotFoundError: No module named 'dehaze'`.

Corrected form:

```bash
PYTHONPATH=src /sda/home/wangyuxin/ConvIR-B/envs/convir-cu121/bin/python \
  -m unittest discover -s tests -v
```

## 2026-09-15: PowerShell to WSL loop variables

An inline PowerShell double-quoted command containing Bash `"$f"` expanded
`$f` before reaching WSL, so `python -m json.tool` received an empty path.
Use a PowerShell single-quoted here-string, strip CRLF, and pipe the Bash script
body to WSL so Bash owns the loop variable expansion.
