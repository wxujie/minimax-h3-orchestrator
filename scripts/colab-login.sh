#!/usr/bin/env bash
# 登录一个 Colab 账号到它的隔离 HOME（一次性操作）。
# 用法: bash scripts/colab-login.sh <account-id>
# 例:   bash scripts/colab-login.sh colab-account-2
#
# 每个账号只需跑一次。登录后 token 落在
#   <base>/<account-id>/.config/colab-cli/token.json
# controller 的 ColabManager 运行时会自动按账号注入对应 HOME，无需再手动切换。
set -euo pipefail

ACCT="${1:?用法: colab-login.sh <account-id>}"
BASE="${COLAB_ACCOUNTS_HOME:-$HOME/.colab-accounts}"
ACCT_HOME="$BASE/$ACCT"

# 定位项目根目录（脚本在 scripts/ 下）
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLBIN="$PROJ/.venv/bin/colab"

if [[ ! -x "$COLBIN" ]]; then
  echo "找不到 colab 二进制: $COLBIN" >&2
  echo "请先装好依赖: uv pip install google-colab-cli --index-url https://pypi.org/simple/" >&2
  exit 1
fi

mkdir -p "$ACCT_HOME"

echo "登录账号: $ACCT"
echo "隔离 HOME: $ACCT_HOME"
echo "浏览器会弹出授权链接，批准后把 code 粘回来。"
echo "---"

# Colab CLI 的 OAuth scope 校验有 bug（Google 返回的 scope 集合与请求不一致），
# 不放宽会抛 Warning: Scope has changed。这个环境变量让 oauthlib 容忍 scope 差异。
export OAUTHLIB_RELAX_TOKEN_SCOPE=1

HOME="$ACCT_HOME" "$COLBIN" status

echo "---"
echo "完成。token 已写入: $ACCT_HOME/.config/colab-cli/token.json"
echo "现在在 .env 里启用即可: COLAB_ACCOUNT_<N>_ENABLED=true"