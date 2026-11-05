"""Own the local web/MCP processes; never stop unrelated listeners."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / '.task_tmp/local-backend'
SERVICES = {'survey': 8761, 'mask': 8762, 'report': 8763, 'model': 8764, 'video': 8765, 'web': 8780}


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.3):
            return True
    except OSError:
        return False


def supervise():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Bind a dedicated local lock for the lifetime of this supervisor.
    lock = socket.socket()
    lock.bind(('127.0.0.1', 18780))
    lock.listen(1)
    occupied = [port for port in SERVICES.values() if listening(port)]
    if occupied:
        raise RuntimeError(f'Ports already in use: {occupied}; existing processes were not changed.')
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT/'src'), PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               FACADE_CONFIG_FILE=str(ROOT/'.env'), FASTMCP_CHECK_FOR_UPDATES='off')
    children, streams = {}, []
    state = {'session': uuid.uuid4().hex, 'supervisor_pid': os.getpid(), 'status': 'starting',
             'started': time.time(), 'services': {}}

    def publish():
        state['heartbeat'] = time.time()
        tmp = STATE_DIR/'state.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(STATE_DIR/'state.json')

    try:
        publish()
        for name, port in SERVICES.items():
            module = 'facade_mcp.web_app' if name == 'web' else f'facade_mcp.{name}_server'
            args = [sys.executable, '-m', module, '--port', str(port)]
            if name != 'web':
                args += ['--transport', 'http', '--host', '127.0.0.1']
            log = (STATE_DIR/f'{name}.log').open('ab')
            streams.append(log)
            child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=log,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            children[name] = child
            state['services'][name] = {'pid':child.pid,'port':port,
                'url':f'http://127.0.0.1:{port}' + ('/' if name == 'web' else '/mcp'), 'log':str(STATE_DIR/f'{name}.log')}
        while True:
            stop = STATE_DIR/'stop.request'
            if stop.exists() and stop.read_text(encoding='utf-8-sig').strip() == state['session']:
                break
            failed = {n:p.returncode for n,p in children.items() if p.poll() is not None}
            if failed:
                raise RuntimeError(f'Backend child exited: {failed}; see service logs.')
            ready = all(listening(port) for port in SERVICES.values())
            if not ready and time.time()-state['started'] > 90:
                raise RuntimeError('Startup timed out; see service logs.')
            state['status'] = 'running' if ready else 'starting'
            publish()
            time.sleep(2)
    except Exception as error:
        state['error'] = str(error)
    finally:
        shutdown_errors = []
        for child in children.values():
            if child.poll() is None:
                try:
                    if os.name == 'nt':
                        result = subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                            capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                        if result.returncode and child.poll() is None:
                            raise RuntimeError(result.stderr.decode(errors='replace').strip())
                    else:
                        child.terminate()
                    child.wait(timeout=10)
                except Exception as error:
                    shutdown_errors.append(f'PID {child.pid}: {error}')
        for stream in streams:
            stream.close()
        if shutdown_errors:
            state['error'] = '; '.join(shutdown_errors)
        state['status'] = 'failed' if state.get('error') else 'stopped'
        publish()
        lock.close()


if __name__ == '__main__':
    supervise()
