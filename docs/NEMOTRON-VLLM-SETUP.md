# Nemotron standard-vLLM setup

This is the first deployment target for the project: NVIDIA Nemotron 3 Nano
served by the validated Fuzzball vLLM catalog entry. It is a normal vLLM
OpenAI-compatible worker. The gateway and KV-cache experiments come after this
worker can answer a short request.

## Hugging Face authentication

The user-level Hugging Face CLI is installed at:

```text
/Users/sayansharma/.local/bin/hf
```

Log in without putting a token in this repository:

```sh
/Users/sayansharma/.local/bin/hf auth login
/Users/sayansharma/.local/bin/hf auth whoami
```

The public Nemotron checkpoint may download anonymously, but a token avoids Hub
rate limits and is required if access changes. To store the token in Fuzzball,
use the Fuzzball secret store rather than an environment file:

```sh
HF_TOKEN="$(/Users/sayansharma/.local/bin/hf auth token)"
printf '{"token":"%s"}\n' "$HF_TOKEN" \
  | /Users/sayansharma/code/fuzzball-tools/fuzzball secret create \
      secret://group/nemotron-hf --type hf
unset HF_TOKEN
```

Do not commit the token, print it, or put it in a Fuzzball workflow definition.
The token is used only during `hf://` ingress; the serving container uses the
downloaded local files.

## Standard vLLM worker

The venue catalog entry is:

```text
Nemotron 3 Nano
```

It serves:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

Start one replica for the first smoke test:

```sh
fuzzball workflow catalog start "Nemotron 3 Nano" \
  --name ss-nemotron-bf16-smoke \
  --values Model=hf://nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16,MaxReplicas=1 \
  --watch
```

If a token is needed, use the catalog's `HfTokenSecret` value when prompted or
passed through `--values`:

```sh
--values HfTokenSecret=secret://group/nemotron-hf
```

Keep the first run on the default ephemeral model volume. Once the endpoint is
working, create a dedicated writable Team 2 model volume and set `Volume` to
that volume so later restarts do not redownload the checkpoint. Do not reuse
the shared `datasets` volume unless its ownership and write permissions have
been explicitly fixed; the first attempt failed because its model directory
was not writable.

## Verify the worker

After the workflow becomes ready, list its endpoint:

```sh
fuzzball workflow endpoints list
```

Then verify the served model and one short streaming request through the
endpoint's OpenAI-compatible URL:

```sh
curl "$WORKER_URL/v1/models"
curl -N "$WORKER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", \
       "messages":[{"role":"user","content":"Reply with one word: ready"}], \
       "max_tokens":16,"temperature":0,"stream":true}'
```

Only after this succeeds should the endpoint be added to the gateway as a
worker origin. This catalog worker is BF16 model weights; it is not yet the
FP8/BF16 KV-cache comparison.
