#!/usr/bin/env bash
# create-worker-tunnel.sh — 本地自动化建 Cloudflare locally-managed 隧道，
# 并把 credentials + config 写回 .env 对应账号字段。
#
# 用法:
#   ./scripts/create-worker-tunnel.sh <account_id> <notebook_name> <gpu_count> <agent_port_base> <tunnel_domain>
#
# 例（kaggle-account-1 单卡，agent 监听 8000）:
#   ./scripts/create-worker-tunnel.sh kaggle-account-1 nb-kaggle-account-1 1 8000 tunnel.jayapp.cn
#
# 前置条件：本机已 `cloudflared tunnel login`（存在 ~/.cloudflared/cert.pem），
# 且 tunnel_domain 的 zone 已托管在 Cloudflare。
#
# 行为：
#   1) 若 account 在 .env 里已有 TUNNEL_CONFIG/TUNNEL_CREDENTIALS，跳过（幂等）。
#   2) 否则为每个 gpu worker 建隧道 + 配 DNS + 提取凭证 + 生成 config.yml。
#   3) 把多 worker 的 config 合并后写回 .env（字段名按 account 类型）。
#
# 产物也落在 scripts/.tunnels/<worker_id>/ 便于排查。
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "用法: $0 <account_id> <notebook_name> <gpu_count> <agent_port_base> <tunnel_domain>" >&2
  echo "例:   $0 kaggle-account-1 nb-kaggle-account-1 1 8000 tunnel.jayapp.cn" >&2
  exit 2
fi

ACCOUNT_ID="$1"
NOTEBOOK_NAME="$2"
GPU_COUNT="$3"
AGENT_PORT_BASE="$4"
TUNNEL_DOMAIN="$5"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
OUT_ROOT="$SCRIPT_DIR/.tunnels"

# cloudflared 必须在 PATH 或已下载
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "❌ 未找到 cloudflared，请先安装" >&2
  exit 1
fi

# 登录检查（cert.pem）
CERT="${HOME}/.cloudflared/cert.pem"
if [ ! -f "$CERT" ]; then
  echo "❌ 未找到 ${CERT}" >&2
  echo "   请先在本机执行: cloudflared tunnel login" >&2
  exit 1
fi

# 根据 account_id 推导 .env 字段前缀
case "$ACCOUNT_ID" in
  kaggle-account-*)
    PREFIX=$(echo "$ACCOUNT_ID" | sed 's/kaggle-account-/KAGGLE_ACCOUNT_/')
    ;;
  colab-account-*)
    PREFIX=$(echo "$ACCOUNT_ID" | sed 's/colab-account-/COLAB_ACCOUNT_/')
    ;;
  *)
    echo "❌ 无法识别 account_id: $ACCOUNT_ID（应为 kaggle-account-N 或 colab-account-N）" >&2
    exit 2
    ;;
esac
CONFIG_KEY="${PREFIX}_TUNNEL_CONFIG"
CREDS_KEY="${PREFIX}_TUNNEL_CREDENTIALS"

# 已在 .env 配好则跳过（幂等）
if grep -q "^${CONFIG_KEY}=.\+" "$ENV_FILE" 2>/dev/null \
   && grep -q "^${CREDS_KEY}=.\+" "$ENV_FILE" 2>/dev/null; then
  echo "✅ ${ACCOUNT_ID} 已在 .env 配置隧道，跳过"
  exit 0
fi

echo "🔧 为 ${ACCOUNT_ID} 生成隧道（${GPU_COUNT} worker）..."

configs=""
creds=""
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  WORKER_ID="${NOTEBOOK_NAME}-gpu${gpu}"
  SUBDOMAIN="${WORKER_ID}"
  AGENT_PORT=$((AGENT_PORT_BASE + gpu))
  TUNNEL_NAME="worker-${WORKER_ID}"
  HOSTNAME="${SUBDOMAIN}.${TUNNEL_DOMAIN}"
  OUT_DIR="${OUT_ROOT}/${WORKER_ID}"
  mkdir -p "$OUT_DIR"

  # 建隧道（幂等）
  if cloudflared tunnel info "$TUNNEL_NAME" >/dev/null 2>&1; then
    echo "  ✅ 隧道 '${TUNNEL_NAME}' 已存在，复用"
  else
    echo "  🆕 创建隧道 '${TUNNEL_NAME}'"
    cloudflared tunnel create "$TUNNEL_NAME" >/dev/null
  fi
  TUNNEL_ID="$(cloudflared tunnel info "$TUNNEL_NAME" --output json | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')"

  # 配 DNS
  echo "  🌐 配置 DNS: ${HOSTNAME}"
  cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$HOSTNAME" >/dev/null

  # 提取 credentials
  CRED_SRC="${HOME}/.cloudflared/${TUNNEL_ID}.json"
  CRED_DST="${OUT_DIR}/credentials.json"
  cp "$CRED_SRC" "$CRED_DST"

  # 生成 config.yml
  cat > "${OUT_DIR}/config.yml" <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: /tmp/cloudflared/credentials.json

ingress:
  - hostname: ${HOSTNAME}
    service: http://localhost:${AGENT_PORT}
  - service: http_status:404
EOF

  # 累计进合并变量
  configs+="$(cat "${OUT_DIR}/config.yml")"$'\n'
  creds+="$(cat "$CRED_DST" | tr -d '\n')"$'\n'
  echo "  ✅ ${WORKER_ID} -> https://${HOSTNAME}"
done

# 写回 .env（用 python 安全处理多行 + 特殊字符）
python3 - "$ENV_FILE" "$CONFIG_KEY" "$CREDS_KEY" "$configs" "$creds" <<'PY'
import sys, re
env_file, config_key, creds_key, configs, creds = sys.argv[1:6]
configs = configs.rstrip("\n")
creds = creds.rstrip("\n")

def set_env_var(path, key, value):
    # 值里有换行/引号，写入时用单引号包裹并转义内部单引号（.env 兼容 python-dotenv）
    esc = value.replace("'", "'\\''")
    line = f"{key}='{esc}'\n"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        content = ""
    # 多行值：匹配 KEY= 到下一个 ^[A-Z_]+= 或文件尾（DOTALL 跨行）
    pattern = re.compile(rf"^{key}=.*?(?=^[A-Z_][A-Z0-9_]*=|\Z)",
                         flags=re.M | re.DOTALL)
    if pattern.search(content):
        content = pattern.sub(line, content)
    else:
        content = content.rstrip("\n") + "\n" + line
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

set_env_var(env_file, config_key, configs)
set_env_var(env_file, creds_key, creds)
print(f"✅ 已写回 .env: {config_key} / {creds_key}")
PY

echo
echo "完成。下次启动 ${ACCOUNT_ID} 的 session 时，controller 会自动注入固定隧道配置。"