import subprocess, os, time, glob, shutil, sys

TOTAL = 9053
PART = int(os.environ.get('PART_SIZE', '600'))
starts = list(range(0, TOTAL, PART))

def part_ok(s):
    out = 'khanh_part_%05d.mp4' % s
    if not os.path.exists(out) or os.path.getsize(out) < 100000:
        return False
    try:
        r = subprocess.run(['ffmpeg', '-i', out], capture_output=True, text=True, timeout=30)
        import re
        m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', r.stderr)
        d = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        return False
    expect = (min(s + PART, TOTAL) - s) / 60.0
    return abs(d - expect) < 0.4

t_all = time.time()
for i, s in enumerate(starts):
    out = 'khanh_part_%05d.mp4' % s
    if part_ok(s):
        print('[%s] part %d/%d (%d-%d) already done, skipping' % (time.strftime('%H:%M:%S'), i + 1, len(starts), s, min(s + PART, TOTAL)), flush=True)
        continue
    ok = False
    for attempt in (1, 2):
        print('[%s] part %d/%d (%d-%d) attempt %d starting' % (time.strftime('%H:%M:%S'), i + 1, len(starts), s, min(s + PART, TOTAL), attempt), flush=True)
        t0 = time.time()
        with open('part_%05d.log' % s, 'w') as lf:
            p = subprocess.run(['python3', '-u', 'part_worker.py', str(s)], stdout=lf, stderr=subprocess.STDOUT)
        el = time.time() - t0
        if p.returncode == 0 and part_ok(s):
            ok = True
            print('[%s] part %d (%d-%d) OK in %.0fs' % (time.strftime('%H:%M:%S'), s, s, min(s + PART, TOTAL), el), flush=True)
            break
        print('[%s] part %d (%d-%d) FAILED rc=%d after %.0fs — retrying after cleanup' % (time.strftime('%H:%M:%S'), s, s, min(s + PART, TOTAL), p.returncode, el), flush=True)
        for junk in glob.glob('/tmp/ort.quant.*'):
            shutil.rmtree(junk, ignore_errors=True)
        time.sleep(10)
    if not ok:
        print('ABORT: part %d failed twice' % s, flush=True)
        sys.exit(1)

# concat all parts (video only)
print('[%s] all parts done, concatenating...' % time.strftime('%H:%M:%S'), flush=True)
with open('concat_list.txt', 'w') as f:
    for s in starts:
        f.write("file 'khanh_part_%05d.mp4'\n" % s)
r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', 'concat_list.txt', '-c', 'copy', 'khanh_video_only.mp4'], capture_output=True, text=True)
if r.returncode != 0:
    print('concat failed:', r.stderr, flush=True)
    sys.exit(2)
print('[%s] video concat done, muxing audio...' % time.strftime('%H:%M:%S'), flush=True)
r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', 'khanh_video_only.mp4', '-i', 'short.webm',
                    '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy', '-shortest', '-movflags', '+faststart',
                    'khanh_final.mp4'], capture_output=True, text=True)
if r.returncode != 0:
    print('mux failed:', r.stderr, flush=True)
    sys.exit(3)
print('[%s] ALL DONE -> khanh_final.mp4 total %.0fs' % (time.strftime('%H:%M:%S'), time.time() - t_all), flush=True)
