import os, sys

def drop(path):
    try:
        f = open(path, 'rb')
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            f.close()
        return 1
    except Exception:
        return 0

total = 0
if len(sys.argv) > 1 and sys.argv[1] == 'large':
    # drop all files > 5MB in given roots (except models dir)
    roots = ['/usr/local/lib/python3.13/site-packages', '/home/user/.deno', '/tmp', '/home/user/video_swap/preview', '/home/user/uploads', '/home/user/video_swap/test']
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                p = os.path.join(dirpath, fn)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > 5 * 1024 * 1024:
                    total += drop(p)
else:
    for p in sys.argv[1:]:
        total += drop(p)
print('dropped %d files' % total, flush=True)
