# Source this on the machine or node you work on:  source env.sh
#
# Reads config.env, the paths and cluster settings of this checkout (gitignored; see config.example.env), then builds
# and activates the environment of the whole repository from pyproject.toml and uv.lock: omop_etl, the cohort builder,
# its tests and the notebook (PyPOTS on CPU torch, Jupyter). Jobs submitted from this shell run this checkout
# (ICU_OMOP_REPO).
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export ICU_OMOP_REPO="$PROJECT_DIR"
if [ -f "$PROJECT_DIR/config.env" ]; then
    source "$PROJECT_DIR/config.env"
fi
unset VIRTUAL_ENV
ENV_DIR="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/.venv}"
mkdir -p "$(dirname "$ENV_DIR")"
# the lock serialises concurrent jobs syncing the same environment
if flock "$(dirname "$ENV_DIR")/.icu-omop-cohorts.sync.lock" uv sync --quiet --project "$PROJECT_DIR"; then
    source "$ENV_DIR/bin/activate"
    # a .venv link here makes editors that scan the project find the environment
    [ "$ENV_DIR" != "$PROJECT_DIR/.venv" ] && ln -sfn "$ENV_DIR" "$PROJECT_DIR/.venv"
    echo "  environment ready: $ENV_DIR/bin/python"
else
    echo "uv sync failed - run 'uv sync --project $PROJECT_DIR' to see why" >&2
fi
