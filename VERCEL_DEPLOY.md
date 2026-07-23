# Deploy AgriShield ML to Vercel (Development)

## Prerequisites

1. [Vercel CLI](https://vercel.com/docs/cli) installed (`npm i -g vercel`)
2. ONNX model file (`best.onnx` or `best 2.onnx`)

## Option A — Bundle model in repo

```bash
mkdir -p models
# Copy your ONNX model:
cp /path/to/best.onnx models/best.onnx
```

## Option B — Download model at runtime

Set env var in Vercel project settings:

```
MODEL_URL=https://your-server.com/path/to/best.onnx
```

## Deploy

From this `vercel/` folder:

```bash
cd Proto1/agri_shield-master/vercel
vercel login
vercel --prod
```

Note the deployment URL (e.g. `https://vercel-sage-six-77.vercel.app`).

Production alias (current): **https://vercel-sage-six-77.vercel.app**

## Test

```bash
curl https://YOUR-URL.vercel.app/health
curl -X POST https://YOUR-URL.vercel.app/detect -F "image=@test.jpg"
```

## Wire PHP

In `Proto1/flask_api_config.php`, set:

```php
$FLASK_ENV = 'vercel';
```

And update the `vercel` URL with your deployment URL.

## Limits

- Cold starts: first request may take 5–15 seconds
- Training (`/train`) is **not** on Vercel — use Google Colab
- Max function duration depends on your Vercel plan
