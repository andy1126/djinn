#!/usr/bin/env bash
# 开发环境一键启动 / 停止 / 重启前后端。
#
# 用法:
#   ./scripts/dev.sh start      启动前后端(已运行则跳过)
#   ./scripts/dev.sh stop       停止前后端
#   ./scripts/dev.sh restart    重启前后端
#   ./scripts/dev.sh status     查看运行状态

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_LOG="$ROOT/.cache/backend.log"
FRONTEND_LOG="$ROOT/.cache/frontend.log"

mkdir -p "$ROOT/.cache"

say()  { printf '\033[32m[dev]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[dev]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[dev]\033[0m %s\n' "$*" >&2; exit 1; }

# 监听某端口的 PID 列表(空 = 未运行)。
# 后端 --reload 的 reloader 与 worker 都在监听端口,用端口定位才能两者一起停掉。
port_pids() { lsof -tnP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null; }

stop_port() {
  local port="$1" pids
  pids="$(port_pids "$port")"
  [ -n "$pids" ] || return 1
  # 先优雅停(SIGTERM),等退出;仍占用再强制杀。
  echo "$pids" | xargs kill 2>/dev/null || true
  for _ in $(seq 1 20); do
    [ -z "$(port_pids "$port")" ] && return 0
    sleep 0.25
  done
  pids="$(port_pids "$port")"
  [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
  return 0
}

start_backend() {
  if [ -n "$(port_pids "$BACKEND_PORT")" ]; then
    warn "后端已在运行(http://localhost:$BACKEND_PORT)"
    return
  fi
  [ -x "$ROOT/.venv/bin/python" ] || die "未找到 .venv,请先 uv pip install -e \".[dev]\""
  nohup .venv/bin/python -m uvicorn djinn.api.main:app \
    --host 0.0.0.0 --port "$BACKEND_PORT" --reload \
    > "$BACKEND_LOG" 2>&1 &
  say "后端启动中 → http://localhost:$BACKEND_PORT(日志 $BACKEND_LOG)"
}

start_frontend() {
  if [ -n "$(port_pids "$FRONTEND_PORT")" ]; then
    warn "前端已在运行(http://localhost:$FRONTEND_PORT)"
    return
  fi
  [ -x "$ROOT/frontend/node_modules/.bin/vite" ] || die "未找到 vite,请先在 frontend 下 npm install"
  nohup bash -c "cd '$ROOT/frontend' && exec ./node_modules/.bin/vite --port $FRONTEND_PORT" \
    > "$FRONTEND_LOG" 2>&1 &
  say "前端启动中 → http://localhost:$FRONTEND_PORT(日志 $FRONTEND_LOG)"
}

stop_backend() {
  if stop_port "$BACKEND_PORT"; then
    say "后端已停止"
  else
    warn "后端未在运行"
  fi
}

stop_frontend() {
  if stop_port "$FRONTEND_PORT"; then
    say "前端已停止"
  else
    warn "前端未在运行"
  fi
}

status() {
  [ -n "$(port_pids "$BACKEND_PORT")" ] \
    && say "后端: 运行中 → http://localhost:$BACKEND_PORT" \
    || warn "后端: 未运行"
  [ -n "$(port_pids "$FRONTEND_PORT")" ] \
    && say "前端: 运行中 → http://localhost:$FRONTEND_PORT" \
    || warn "前端: 未运行"
}

case "${1:-}" in
  start)   start_backend; start_frontend; sleep 1; status ;;
  stop)    stop_backend;  stop_frontend ;;
  restart) stop_backend; stop_frontend; sleep 1; start_backend; start_frontend; sleep 1; status ;;
  status)  status ;;
  *) die "用法: $0 {start|stop|restart|status}" ;;
esac
