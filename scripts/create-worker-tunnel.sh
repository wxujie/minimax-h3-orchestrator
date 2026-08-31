#!/usr/bin/env bash
# create-worker-tunnel.sh — 本地自动化建 Cloudflare locally-managed 隧道。
#
# 用法:
#   ./scripts/create-worker-tunnel.sh <worker_id> <subdomain> <agent_port> <tunnel_domain>
#
# 例（kaggle-account-1 的 gpu0 worker，agent 监听 8000）:
#   ./scripts/create-worker-tunnel.sh nb-kaggle-account-1-gpu0 nb-kaggle-account-1-gpu0 8000 tunnel.jayapp.cn
#
# 前置条件：本机已 `cloudflared tunnel login`（存在 ~/.cloudflared/cert.pem），
# 且 tunnel.jayapp.cn 的 zone 已托管在 Cloudflare（route dns 会自动写 CNAME）。
#
# 产物（都落在 scripts/.tunnels/<worker_id>/）:
#   credentials.json  隧道凭证（等价 token，交给 worker）
#   config.yml        ingress 映射（子域名 -> localhost:<agent_port>）
#
# 结果会以环境变量片段的形式打印，便于贴进 .env。
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "用法: $0 <worker_id> <subdomain> <agent_port> <tunnel_domain>" >&2
  exit 2
fi

WORKER_ID="$1"
SUBDOMAIN="$2"
AGENT_PORT="$3"
TUNNEL_DOMAIN="$4"
TUNNEL_NAME="worker-${WORKER_ID}"
HOSTNAME="${SUBDOMAIN}.${TUNNEL_DOMAIN}"

OUT_DIR="$(cd "$(dirname "$0")" && pwd)/.tunnels/${WORKER_ID}"
mkdir -p "$OUT_DIR"

# 1. 确保已登录（cert.pem 存在），否则给出指引
CERT="${HOME}/.cloudflared/cert.pem"
if [ ! -f "$CERT" ]; then
  echo "❌ 未找到 ${CERT}" >&2
  echo "   请先在本机执行: cloudflared tunnel login" >&2
  exit 1
fi

# 2. 建隧道（幂等：已存在则复用）
if cloudflared tunnel info "$TUNNEL_NAME" >/dev/null 2>&1; then
  echo "✅ 隧道 '${TUNNEL_NAME}' 已存在，复用"
  TUNNEL_ID="$(cloudflared tunnel info "$TUNNEL_NAME" --output json | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"
else
  echo "🆕 创建隧道 '${TUNNEL_NAME}' ..."
  cloudflared tunnel create "$TUNNEL_NAME"
  TUNNEL_ID="$(cloudflared tunnel info "$TUNNEL_NAME" --output json | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"
  echo "   tunnel id: ${TUNNEL_ID}"
fi

# 3. 配 DNS（幂等，--overwrite-dns 可安全重复执行）
echo "🌐 配置 DNS: ${HOSTNAME}"
cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$HOSTNAME"

# 4. 提取 credentials 文件（等价 token，交给 worker）
CRED_SRC="${HOME}/.cloudflared/${TUNNEL_ID}.json"
CRED_DST="${OUT_DIR}/credentials.json"
cp "$CRED_SRC" "$CRED_DST"
echo "🔑 credentials -> ${CRED_DST}"

# 5. 生成 config.yml（ingress 映射到 agent 端口）
cat > "${OUT_DIR}/config.yml" <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: /tmp/cloudflared/credentials.json

ingress:
  - hostname: ${HOSTNAME}
    service: http://localhost:${AGENT_PORT}
  - service: http_status:404
EOF
echo "📝 config -> ${OUT_DIR}/config.yml"

# 6. 打印环境变量片段（贴进 .env）
echo
echo "──────────────────────────────────────────────"
echo " 贴进 .env（按账号对应字段）:"
echo "──────────────────────────────────────────────"
echo "WORKER_TUNNEL_${WORKER_ID}_CREDENTIALS=\"$(cat "$CRED_DST" | tr -d '\n')\""
echo "──────────────────────────────────────────────"
echo " worker_id    : ${WORKER_ID}"
echo " 公共 URL     : https://${HOSTNAME}"
echo " 隧道名       : ${TUNNEL_NAME}"
echo " 产物目录     : ${OUT_DIR}"