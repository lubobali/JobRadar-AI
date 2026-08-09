#!/usr/bin/env bash
#
# Deploy the JobRadar MCP server to jobradar.lubot.ai.
#
#   ssh root@100.115.173.71 'bash -s' < scripts/deploy_mcp.sh
#
# Idempotent: safe to re-run for every deploy. The first run creates the user,
# the venv, the token and the vhost; later runs pull, reinstall and restart.
#
# This host also runs LuBot in production. Everything here is deliberately
# isolated from it - own directory, own unprivileged user, own venv, own port,
# own systemd unit, own nginx vhost. Nothing is shared in either direction.
#
# Why the MCP server lives here and not as a Databricks App: the Databricks AI
# Gateway cannot authenticate to a Databricks App. Proven on the previous
# project against every method its own form offers - DCR (the workspace OIDC
# server publishes no registration_endpoint), PAT (Apps take OAuth only and
# 302 to /authorize), OAuth M2M (needs a service principal, which is
# admin-only), OAuth U2M (needs a redirect URI on a Databricks-managed client).
# So the Databricks App is the FRONTEND, and the MCP server is here.

set -euo pipefail

REPO="https://github.com/lubobali/JobRadar-AI.git"
DIR="/opt/jobradar"
USER="jobradar"
PORT="8403"
DOMAIN="jobradar.lubot.ai"
EMAIL="lubobali23@gmail.com"

echo "==> user and directory"
id "$USER" >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin "$USER"

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch -q origin
  git -C "$DIR" reset -q --hard origin/main
else
  git clone -q "$REPO" "$DIR"
fi
echo "    commit: $(git -C "$DIR" rev-parse --short HEAD)"

echo "==> venv"
test -d "$DIR/venv" || python3 -m venv "$DIR/venv"
"$DIR/venv/bin/pip" install -q --upgrade pip
"$DIR/venv/bin/pip" install -q -r "$DIR/mcp_server/requirements.txt"
"$DIR/venv/bin/pip" install -q -e "$DIR"

echo "==> environment"
# The token is generated ON THIS HOST and never leaves it except into the
# systemd unit that reads it. It is not printed, not committed, and not passed
# through anyone's terminal.
umask 077
if [ ! -f "$DIR/.env" ]; then
  cat > "$DIR/.env" <<EOF
JOBRADAR_BEARER_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
JOBRADAR_USER_EMAIL=data@lubobali.com
PORT=$PORT
HOST=127.0.0.1
LOG_LEVEL=INFO
EOF
  echo "    generated a new bearer token"
else
  echo "    .env already present, left alone"
fi

# LAKEBASE_URL is not stored here. The server reads it from the Databricks
# secret scope at runtime through the same WorkspaceClient path the notebooks
# use, so the connection string exists in exactly one place.
grep -q '^LAKEBASE_URL=' "$DIR/.env" || cat >> "$DIR/.env" <<'EOF'
# Set this if the host has no Databricks credentials of its own. Otherwise
# lakebase.py resolves it from the lubo-jobradar secret scope.
EOF

chown -R "$USER:$USER" "$DIR"
chmod 600 "$DIR/.env"

echo "==> systemd"
cat > /etc/systemd/system/jobradar.service <<EOF
# JobRadar-AI MCP server.
#
# Isolated from everything else on this host: own directory, own venv, own
# unprivileged user, own port, own unit. Binds 127.0.0.1 only; nginx
# terminates TLS in front, so the port is never reachable from outside.
[Unit]
Description=JobRadar-AI MCP server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$DIR/mcp_server
EnvironmentFile=$DIR/.env
ExecStart=$DIR/venv/bin/python $DIR/mcp_server/jobs_mcp_server.py
Restart=on-failure
RestartSec=5

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/tmp

# libpq looks for an OPTIONAL client certificate at ~/.postgresql/postgresql.crt
# on every connection. ProtectHome=true hides /home, so that lookup returns
# "Permission denied" rather than "No such file" - and libpq treats a
# permission error as fatal where it would have shrugged off a missing file:
#
#   could not open certificate file "/home/jobradar/.postgresql/postgresql.crt":
#   Permission denied
#
# Which surfaces as OperationalError with a correct URL, correct credentials,
# and a connection that works from any shell on the same host. Pointing libpq
# at paths that simply do not exist fixes it without weakening the sandbox;
# this connection uses sslmode=require and no client certificate.
Environment=PGSSLCERT=/tmp/no-client.crt
Environment=PGSSLKEY=/tmp/no-client.key

# Same cause, different library. HF_HOME defaults to ~/.cache, which
# ProtectHome=true hides, so sentence-transformers cannot write the model it
# downloads - and search_jobs fails while every non-embedding tool works, which
# points at the search code rather than at the sandbox.
Environment=HF_HOME=/tmp/.cache/huggingface
Environment=HF_HUB_DISABLE_PROGRESS_BARS=1
Environment=TOKENIZERS_PARALLELISM=false

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable -q --now jobradar
systemctl restart jobradar
sleep 4
systemctl is-active jobradar

echo "==> nginx"
cat > /etc/nginx/sites-available/$DOMAIN <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_http_version 1.1;

        proxy_set_header Host              \$host;
        proxy_set_header X-Real-IP         \$remote_addr;
        proxy_set_header X-Forwarded-For   \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Connection        "";

        # MCP streamable HTTP holds the response open and emits events as they
        # happen. Buffering would collect the whole stream before forwarding a
        # byte, which turns a working server into one that appears to hang.
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
EOF
ln -sfn /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/$DOMAIN
nginx -t
systemctl reload nginx

echo "==> TLS"
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL" --redirect
else
  echo "    certificate already present"
fi

echo "==> checks"
curl -s -o /dev/null -w "    /status  -> %{http_code}\n" "http://127.0.0.1:$PORT/status"
curl -s -o /dev/null -w "    /mcp     -> %{http_code} (401 expected)\n" "http://127.0.0.1:$PORT/mcp"
echo
echo "Deployed. The bearer token is in $DIR/.env and has not been printed."
echo "To read it:  grep JOBRADAR_BEARER_TOKEN $DIR/.env | cut -d= -f2"
