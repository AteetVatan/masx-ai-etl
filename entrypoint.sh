#!/bin/bash
set -e

MODEL_CACHE_DIR="${CACHE_DIR}/models"

echo "Checking model cache in $MODEL_CACHE_DIR"

micromamba run -n appenv python /app/init_models.py


# if [ ! -d "$MODEL_CACHE_DIR/models--facebook--bart-large-cnn" ]; then
#     echo "Preloading Hugging Face models..."
#     micromamba run -n appenv python /app/init_models.py
# else
#     echo "Models already cached"
# fi

# Start your app
exec micromamba run -n appenv python -u /app/handler.py
