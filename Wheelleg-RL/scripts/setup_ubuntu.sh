#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Mirror selection:
#   bash setup_ubuntu.sh                 # benchmark and choose the fastest mirror
#   bash setup_ubuntu.sh --interactive   # choose APT/PyPI mirrors manually
#   WHEELLEG_MIRROR=ustc bash ...        # force one mirror for both services
#   WHEELLEG_APT_MIRROR=aliyun ...       # override APT only
#   WHEELLEG_PYPI_MIRROR=tuna ...        # override PyPI only
MODE="auto"
for arg in "$@"; do
  case "$arg" in
    --interactive|-i) MODE="interactive" ;;
    --auto|-a) MODE="auto" ;;
    --help|-h)
      sed -n '1,22p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ "$MODE" == "interactive" && ! -t 0 ]]; then
  echo "--interactive requires a terminal; falling back to automatic selection." >&2
  MODE="auto"
fi

CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"

# Keep URLs without a trailing slash so they can be reused for all endpoints.
declare -A MIRROR_LABEL=(
  [tuna]="清华 TUNA"
  [aliyun]="阿里云"
  [ustc]="中科大 USTC"
  [tencent]="腾讯云"
  [huawei]="华为云"
  [bfsu]="北外 BFSU"
  [sjtu]="上海交大 SJTUG"
  [nju]="南京大学 NJU"
)
declare -A APT_BASE=(
  [tuna]="https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
  [aliyun]="https://mirrors.aliyun.com/ubuntu"
  [ustc]="https://mirrors.ustc.edu.cn/ubuntu"
  [tencent]="https://mirrors.cloud.tencent.com/ubuntu"
  [huawei]="https://repo.huaweicloud.com/repository/ubuntu"
  [bfsu]="https://mirrors.bfsu.edu.cn/ubuntu"
  [sjtu]="https://mirror.sjtu.edu.cn/ubuntu"
  [nju]="https://mirrors.nju.edu.cn/ubuntu"
)
declare -A PYPI_BASE=(
  [tuna]="https://pypi.tuna.tsinghua.edu.cn/simple"
  [aliyun]="https://mirrors.aliyun.com/pypi/simple"
  [ustc]="https://mirrors.ustc.edu.cn/pypi/simple"
  [tencent]="https://mirrors.cloud.tencent.com/pypi/simple"
  [huawei]="https://repo.huaweicloud.com/repository/pypi/simple"
  [bfsu]="https://mirrors.bfsu.edu.cn/pypi/web/simple"
  [sjtu]="https://mirror.sjtu.edu.cn/pypi/web/simple"
  [nju]="https://mirrors.nju.edu.cn/pypi/web/simple"
)
MIRRORS=(tuna aliyun ustc tencent huawei bfsu sjtu nju)

probe() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -L -sS -o /dev/null --connect-timeout 2 --max-time 6 -w '%{time_total}' "$url" 2>/dev/null || echo 999
  elif command -v wget >/dev/null 2>&1; then
    local start end
    start=$(date +%s%N)
    if ! wget -q --timeout=6 --tries=1 -O /dev/null "$url"; then
      echo 999
      return 0
    fi
    end=$(date +%s%N)
    awk "BEGIN { printf \"%.3f\", ($end-$start)/1000000000 }"
  else
    echo 999
  fi
}

choose_interactive() {
  local kind="$1"; shift
  local -a names=("${MIRRORS[@]}")
  echo "请选择 ${kind} 镜像："
  local i=1 name
  for name in "${names[@]}"; do
    echo "  $i) ${MIRROR_LABEL[$name]} (${name})"
    ((i++))
  done
  echo "  0) 自动测速"
  local choice
  read -r -p "输入编号 [0-${#names[@]}]: " choice
  if [[ "$choice" =~ ^[1-8]$ ]]; then
    REPLY="${names[$((choice-1))]}"
  else
    REPLY=""
  fi
}

choose_fastest() {
  local kind="$1"; shift
  local -a names=("${MIRRORS[@]}")
  local best="" best_time=999 name t
  echo "正在测速 ${kind} 镜像（每个最多 6 秒）..."
  for name in "${names[@]}"; do
    if [[ "$kind" == "APT" ]]; then t=$(probe "${APT_BASE[$name]}/dists/${CODENAME}/InRelease"); else t=$(probe "${PYPI_BASE[$name]}/setuptools/"); fi
    printf '  %-16s %ss\n' "${MIRROR_LABEL[$name]}" "$t"
    if awk "BEGIN { exit !($t < $best_time) }"; then best="$name"; best_time="$t"; fi
  done
  REPLY="$best"
  [[ -n "$REPLY" ]] || REPLY="tuna"
}

select_mirror() {
  local kind="$1" forced="${2:-}"
  if [[ -n "$forced" ]]; then
    [[ -n "${MIRROR_LABEL[$forced]:-}" ]] || { echo "未知镜像: $forced" >&2; exit 2; }
    REPLY="$forced"
  elif [[ "$MODE" == "interactive" ]]; then
    choose_interactive "$kind"
    [[ -n "$REPLY" ]] || choose_fastest "$kind"
  else
    choose_fastest "$kind"
  fi
}

select_mirror APT "${WHEELLEG_APT_MIRROR:-${WHEELLEG_MIRROR:-}}"
APT_MIRROR="$REPLY"
select_mirror PyPI "${WHEELLEG_PYPI_MIRROR:-${WHEELLEG_MIRROR:-}}"
PYPI_MIRROR="$REPLY"
APT_URL="${APT_BASE[$APT_MIRROR]}"
PYPI_URL="${PYPI_BASE[$PYPI_MIRROR]}"
echo "APT 使用：${MIRROR_LABEL[$APT_MIRROR]}"
echo "PyPI 使用：${MIRROR_LABEL[$PYPI_MIRROR]}"

# Write a selected source file and use it exclusively for this installation.
# The original APT configuration is preserved and can be used normally afterward.
APT_FILE="/etc/apt/sources.list.d/wheelleg-mirror.list"
APT_CONTENT="deb ${APT_URL} ${CODENAME} main restricted universe multiverse
deb ${APT_URL} ${CODENAME}-updates main restricted universe multiverse
deb ${APT_URL} ${CODENAME}-backports main restricted universe multiverse
deb ${APT_URL} ${CODENAME}-security main restricted universe multiverse
"
printf '%s' "$APT_CONTENT" | sudo tee "$APT_FILE" >/dev/null
APT_OPTIONS=(-o Dir::Etc::sourcelist="$APT_FILE" -o Dir::Etc::sourceparts="-" -o Acquire::Retries=2)
sudo apt-get "${APT_OPTIONS[@]}" update
sudo DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTIONS[@]}" install -y build-essential git curl libgl1-mesa-glx libegl1

if ! command -v uv >/dev/null 2>&1; then
  echo "uv 未安装，使用官方安装脚本。"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# UV_INDEX_URL overrides the default PyPI index without changing the committed pyproject.toml.
export UV_INDEX_URL="$PYPI_URL"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
uv sync
uv run list-envs
