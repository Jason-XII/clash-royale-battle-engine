"""Build the optional observation helpers for this Python interpreter; no third-party build dependency."""
from pathlib import Path
import shlex
import subprocess
import sysconfig

root = Path(__file__).resolve().parent
target = root / ('_native_features' + sysconfig.get_config_var('EXT_SUFFIX'))
command = shlex.split(sysconfig.get_config_var('LDSHARED'))
subprocess.run(command + ['-O3', '-fPIC', '-I' + sysconfig.get_path('include'),
                         str(root / 'native_features.c'), '-o', str(target)], check=True)
print(target)
