import shlex
import subprocess
from pathlib import Path
import os

import modal

streamlit_script_local_path = Path(__file__).parent / "streamlit_run.py"
streamlit_script_remote_path = "/root/streamlit_run.py"

image = (
    modal.Image.debian_slim(python_version="3.9")
    .uv_pip_install("streamlit", "supabase", "pandas", "plotly")
    .env({"FORCE_REBUILD": "true"}) 
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
)

secret = modal.Secret.from_name("custom-secret")

app = modal.App(
    name="usage-dashboard",
    image=image,
    secrets=[secret], 
)

@app.function(allow_concurrent_inputs=100)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = (
        f"streamlit run {target} "
        f"--server.port 8000 "
        f"--server.enableCORS=false "
        f"--server.enableXsrfProtection=false"
    )

    env_vars = {}
    if os.getenv("SUPABASE_KEY"):
        env_vars["SUPABASE_KEY"] = os.getenv("SUPABASE_KEY")
    if os.getenv("SUPABASE_URL"):
        env_vars["SUPABASE_URL"] = os.getenv("SUPABASE_URL")

    subprocess.Popen(cmd, shell=True, env=env_vars)
