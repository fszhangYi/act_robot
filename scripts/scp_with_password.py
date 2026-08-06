"""pexpect-driven scp: feeds the password from env var SCP_PASSWORD, accepts
new host keys automatically. Streams scp's own stdout to keep the progress
visible.

Usage:
    SCP_PASSWORD=... python scp_with_password.py \\
        --src 'huweiwen@10.26.52.122:D:/100_15.7z' \\
        --dst /media/znyyb/EE223AE5223AB287/100_15.7z
"""
from __future__ import annotations

import argparse
import os
import sys

import pexpect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--timeout', type=int, default=3600,
                        help='inactivity timeout per pexpect read (sec)')
    args = parser.parse_args()

    password = os.environ.get('SCP_PASSWORD')
    if not password:
        print('ERROR: SCP_PASSWORD env var not set', file=sys.stderr)
        return 2

    cmd = [
        'scp',
        '-o', 'StrictHostKeyChecking=accept-new',
        '-o', 'UserKnownHostsFile=/dev/null',
        '-o', 'LogLevel=ERROR',
        args.src, args.dst,
    ]
    print(f'[scp] {" ".join(cmd)}', flush=True)

    child = pexpect.spawn(cmd[0], cmd[1:], timeout=args.timeout, encoding='utf-8')
    # Stream child stdout to our stdout in real time
    child.logfile_read = sys.stdout

    try:
        idx = child.expect([
            r'(?i)password:',                         # password prompt
            r'(?i)(yes/no|fingerprint).*',            # host key prompt fallback
            pexpect.EOF,
            pexpect.TIMEOUT,
        ])
        if idx == 0:
            child.sendline(password)
        elif idx == 1:
            child.sendline('yes')
            child.expect(r'(?i)password:')
            child.sendline(password)
        elif idx == 2:
            print('\n[scp] child exited before password prompt', file=sys.stderr)
            child.close()
            return child.exitstatus or 1
        else:
            print('\n[scp] timeout waiting for password prompt', file=sys.stderr)
            child.close(force=True)
            return 3

        child.expect(pexpect.EOF)
        child.close()
        rc = child.exitstatus
        print(f'\n[scp] done (exit={rc})')
        return rc or 0
    except pexpect.exceptions.ExceptionPexpect as exc:
        print(f'\n[scp] pexpect error: {exc}', file=sys.stderr)
        child.close(force=True)
        return 4


if __name__ == '__main__':
    sys.exit(main())
