"""Local sensitivity dashboard. Run: python -B analysis/sensitivity/dashboard.py"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import csv
import json
import re
import subprocess
import sys
import threading

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / 'results'
PROFILE_RE = re.compile(r'P\d{5}')
lock = threading.Lock()
job = {'state': 'idle', 'profile': None, 'message': ''}


def profiles():
    with (ROOT / 'Data/synthetic-profiles.csv').open(newline='', encoding='utf-8') as f:
        return [{'id': r['profile_id'], 'name': r['display_name'], 'status': r['employment_status']}
                for r in csv.DictReader(f)]


def result_file(profile):
    path = RESULTS / 'profiles' / profile / 'sensitivity.json'
    if not path.exists() and profile == 'P00001':
        path = RESULTS / 'alexia_sensitivity.json'
    return path


def worker(profile):
    global job
    try:
        p = subprocess.run([sys.executable, '-B', str(HERE / 'run_sensitivity.py'), '--profile', profile],
                           cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with lock:
            job = {'state': 'done' if p.returncode == 0 else 'error', 'profile': profile,
                   'message': '' if p.returncode == 0 else p.stdout[-4000:]}
    except Exception as e:
        with lock:
            job = {'state': 'error', 'profile': profile, 'message': str(e)}


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, data, content_type='application/json; charset=utf-8'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status, obj):
        self.send_data(status, json.dumps(obj, allow_nan=False).encode())

    def do_GET(self):
        if self.path == '/':
            self.send_data(200, (HERE / 'dashboard.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/profiles':
            self.send_json(200, profiles())
        elif self.path == '/api/status':
            with lock:
                self.send_json(200, job)
        elif self.path.startswith('/api/result/'):
            profile = self.path.removeprefix('/api/result/')
            if not PROFILE_RE.fullmatch(profile):
                return self.send_json(400, {'error': 'Invalid profile ID'})
            path = result_file(profile)
            if not path.exists():
                return self.send_json(404, {'error': 'No analysis saved for this profile yet.'})
            self.send_data(200, path.read_bytes())
        else:
            self.send_json(404, {'error': 'Not found'})

    def do_POST(self):
        global job
        if self.path != '/api/run':
            return self.send_json(404, {'error': 'Not found'})
        # Require an origin matching this loopback server and JSON, preventing cross-site form submissions.
        if self.headers.get('Origin') != f'http://{self.headers.get("Host")}' or self.headers.get('Content-Type') != 'application/json':
            return self.send_json(403, {'error': 'Request must come from this local dashboard'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024: raise ValueError('Invalid request size')
            profile = json.loads(self.rfile.read(length)).get('profile', '')
            if not isinstance(profile, str) or not PROFILE_RE.fullmatch(profile) or profile not in {p['id'] for p in profiles()}:
                raise ValueError('Unknown profile ID')
        except (ValueError, TypeError, json.JSONDecodeError):
            return self.send_json(400, {'error': 'Invalid profile request'})
        with lock:
            if job['state'] == 'running':
                return self.send_json(409, {'error': 'An analysis is already running.'})
            job = {'state': 'running', 'profile': profile, 'message': ''}
            threading.Thread(target=worker, args=(profile,), daemon=True).start()
        self.send_json(202, job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == '__main__': main()
