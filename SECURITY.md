# Security policy

## Project status

Tree Counter v0.1.x is experimental and under active development. Security
fixes are provided for the latest published version only. The plugin executes
locally with the current user's permissions; it is not an operating-system
sandbox.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Contact the maintainer privately through the email address listed in the plugin metadata, including a description, reproduction steps, affected versions, and any mitigation you know.

Do not send raster imagery, model files, credentials, or other sensitive data
with a report. The plugin processes imagery and models locally; it does not
upload models or rasters, collect telemetry or analytics, or upload crash
reports automatically. Network access occurs only after the user explicitly
starts an install, update, or repair action in Runtime Manager.

PyTorch `.pt` checkpoints can execute code while loading. Use only checkpoints
you created or obtained from a trusted source. Tree Counter requires explicit
confirmation of each new checkpoint SHA-256, but this is a trust boundary, not
a malware scanner or sandbox. ONNX is recommended when available.
