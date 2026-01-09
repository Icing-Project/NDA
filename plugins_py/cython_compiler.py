"""
Cython Auto-Compiler for NDA Python Plugins

Provides transparent compilation of Python plugins to native extensions
with SHA256-based cache invalidation and graceful fallback.

Usage:
    The plugin_loader automatically uses this module when available.
    Plugins are compiled on first load and cached for subsequent loads.

Requirements:
    - Cython >= 3.0.0
    - C compiler (gcc on Linux, MSVC on Windows)
    - NumPy (for include headers)

If Cython is not available, the module gracefully returns None and
plugins fall back to pure Python execution.
"""

import hashlib
import platform
import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Logger - warnings only, no tracebacks for compilation failures
logger = logging.getLogger('nda.cython_compiler')

# Cache directory name (relative to plugins_py)
CACHE_DIR_NAME = '.cython_cache'

# Check if Cython is available at import time
try:
    from Cython.Build import cythonize as _cythonize_check
    CYTHON_INSTALLED = True
    del _cythonize_check
except ImportError:
    CYTHON_INSTALLED = False


def is_cython_available() -> bool:
    """Check if Cython compilation is available."""
    return CYTHON_INSTALLED


def get_cache_dir(plugin_dir: Path) -> Path:
    """
    Get or create the cache directory for compiled plugins.

    Args:
        plugin_dir: Path to the plugins_py directory

    Returns:
        Path to the .cython_cache directory
    """
    cache_dir = plugin_dir / CACHE_DIR_NAME
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a plugin file.

    Args:
        file_path: Path to the Python source file

    Returns:
        64-character hex string of SHA256 hash
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_platform_tag() -> tuple:
    """
    Get platform-specific extension and tag for compiled modules.

    Returns:
        Tuple of (extension, platform_tag)
        e.g., ('so', 'x86_64-linux-gnu') or ('pyd', 'win_amd64')
    """
    if platform.system() == 'Windows':
        ext = 'pyd'
        if platform.machine() == 'AMD64':
            plat_tag = 'win_amd64'
        else:
            plat_tag = 'win32'
    else:
        ext = 'so'
        # Linux platform tag
        machine = platform.machine()
        plat_tag = f"{machine}-linux-gnu"

    return ext, plat_tag


def get_compiled_path(plugin_path: Path, cache_dir: Path) -> Path:
    """
    Get the expected path for a compiled plugin.

    Format: {hash_prefix}_{plugin}.cpython-{version}-{platform}.{ext}
    Example: a1b2c3d4e5f6_sine_wave_source.cpython-311-x86_64-linux-gnu.so

    Args:
        plugin_path: Path to the .py plugin file
        cache_dir: Directory to store compiled output

    Returns:
        Path where the compiled module should be stored
    """
    file_hash = compute_file_hash(plugin_path)

    # Get Python version tag
    py_version = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

    # Get platform tag
    ext, plat_tag = get_platform_tag()

    # Use first 12 chars of hash for reasonable uniqueness + readability
    filename = f"{file_hash[:12]}_{plugin_path.stem}.{py_version}-{plat_tag}.{ext}"
    return cache_dir / filename


def is_cache_valid(plugin_path: Path, compiled_path: Path) -> bool:
    """
    Check if compiled cache is still valid.

    Cache is valid if:
    1. Compiled file exists
    2. Hash prefix in filename matches current source hash

    Args:
        plugin_path: Path to source .py file
        compiled_path: Path to compiled .so/.pyd file

    Returns:
        True if cache is valid and can be loaded
    """
    if not compiled_path.exists():
        return False

    # Extract hash from filename (first part before underscore)
    filename = compiled_path.name
    cached_hash = filename.split('_')[0]

    # Compute current hash
    current_hash = compute_file_hash(plugin_path)

    # Compare hash prefix (12 chars)
    return current_hash.startswith(cached_hash)


def get_cython_directives() -> dict:
    """
    Return Cython compiler directives optimized for audio DSP.

    These directives provide maximum performance while maintaining
    NumPy compatibility.

    Returns:
        Dictionary of Cython compiler directives
    """
    return {
        # Safety vs Performance trade-offs (disable for max speed)
        'boundscheck': False,       # Disable bounds checking (MAJOR speedup)
        'wraparound': False,        # Disable negative indexing support
        'cdivision': True,          # Use C division semantics (no ZeroDivisionError)
        'initializedcheck': False,  # Skip memoryview initialization checks

        # Type inference
        'infer_types': True,        # Automatically infer C types

        # Language level
        'language_level': 3,        # Python 3 semantics
    }


def compile_plugin(plugin_path: Path, cache_dir: Path) -> Optional[Path]:
    """
    Compile a Python plugin to a Cython extension.

    Args:
        plugin_path: Path to the .py plugin file
        cache_dir: Directory to store compiled output

    Returns:
        Path to compiled .so/.pyd file, or None if compilation fails

    Note:
        On failure, logs a warning (not full traceback) and returns None.
        Caller should fall back to loading the original Python file.
    """
    try:
        # Import Cython tools - defer to avoid startup overhead if not needed
        from Cython.Build import cythonize
        from setuptools import Extension
        from setuptools.dist import Distribution
        import tempfile
        import shutil

        # Get expected output path
        compiled_path = get_compiled_path(plugin_path, cache_dir)

        # Skip if already compiled and valid
        if is_cache_valid(plugin_path, compiled_path):
            return compiled_path

        # Clean up any old versions of this plugin in cache
        _cleanup_old_versions(cache_dir, plugin_path.stem)

        # Configure extension
        ext = Extension(
            name=plugin_path.stem,
            sources=[str(plugin_path)],
            include_dirs=[],
            language='c',  # Use C backend (simpler, sufficient for our needs)
        )

        # Add NumPy include directory
        try:
            import numpy
            ext.include_dirs.append(numpy.get_include())
        except ImportError:
            pass  # NumPy not available - compilation may still work

        # Get compiler directives
        directives = get_cython_directives()

        # Cythonize (convert .py to C code and compile)
        ext_modules = cythonize(
            [ext],
            compiler_directives=directives,
            quiet=True,  # Suppress compilation output
            force=True,  # Rebuild even if timestamps match (we use hash instead)
        )

        # Build using setuptools
        dist = Distribution({
            'ext_modules': ext_modules,
            'script_name': 'setup.py',
        })

        # Build in temporary directory to avoid polluting source tree
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure build command
            build_ext = dist.get_command_obj('build_ext')
            build_ext.inplace = False
            build_ext.build_lib = tmpdir
            build_ext.build_temp = tmpdir

            # Suppress stdout during build
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                dist.run_command('build_ext')
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Find the built .so/.pyd file
            ext_suffix, _ = get_platform_tag()
            for f in Path(tmpdir).rglob('*'):
                if f.suffix in ('.so', '.pyd') and plugin_path.stem in f.name:
                    # Move to cache with hash-based name
                    shutil.move(str(f), str(compiled_path))
                    return compiled_path

        logger.warning(f"Cython compilation produced no output for {plugin_path.name}")
        return None

    except ImportError:
        # Cython not installed - silent fallback (no warning)
        return None

    except Exception as e:
        # Log warning only (not full traceback per user requirement)
        logger.warning(f"Cython compilation failed for {plugin_path.name}: {type(e).__name__}")
        return None


def _cleanup_old_versions(cache_dir: Path, plugin_stem: str):
    """
    Remove old compiled versions of a specific plugin.

    Called before compiling a new version to prevent stale files.

    Args:
        cache_dir: The .cython_cache directory
        plugin_stem: Plugin name without extension (e.g., 'simple_gain')
    """
    if not cache_dir.exists():
        return

    # Pattern: *_{plugin_stem}.cpython-*
    for cached_file in cache_dir.glob(f'*_{plugin_stem}.cpython-*'):
        try:
            cached_file.unlink()
        except OSError:
            pass  # Ignore errors removing old files


def cleanup_stale_cache(cache_dir: Path, plugin_dir: Path, max_age_days: int = 30):
    """
    Remove old compiled files that no longer match any source plugin.

    Called at startup to keep cache directory clean.

    Args:
        cache_dir: The .cython_cache directory
        plugin_dir: The plugins_py directory
        max_age_days: Remove files older than this many days (default 30)
    """
    import time

    if not cache_dir.exists():
        return

    # Get current source file stems
    source_stems = set()
    for f in plugin_dir.glob('*.py'):
        if not f.name.startswith('_'):
            source_stems.add(f.stem)

    # Also check examples subdirectory
    examples_dir = plugin_dir / 'examples'
    if examples_dir.exists():
        for f in examples_dir.glob('*.py'):
            if not f.name.startswith('_'):
                source_stems.add(f.stem)

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    for cached_file in cache_dir.glob('*'):
        if not cached_file.is_file():
            continue

        should_remove = False

        # Check if too old
        try:
            file_age = current_time - cached_file.stat().st_mtime
            if file_age > max_age_seconds:
                should_remove = True
        except OSError:
            should_remove = True

        # Check if source still exists
        if not should_remove:
            # Extract plugin name from cached filename: {hash}_{name}.cpython-...
            parts = cached_file.name.split('_')
            if len(parts) >= 2:
                # Second part is name.cpython-... so split on .cpython
                name_part = '_'.join(parts[1:])  # Handle names with underscores
                plugin_name = name_part.split('.cpython')[0]
                if plugin_name not in source_stems:
                    should_remove = True

        if should_remove:
            try:
                cached_file.unlink()
            except OSError:
                pass  # Ignore errors removing stale files


def find_cached_module(plugin_path: Path, cache_dir: Path) -> Optional[Path]:
    """
    Find a valid cached compiled module for a plugin.

    Args:
        plugin_path: Path to the .py plugin file
        cache_dir: Directory where compiled modules are stored

    Returns:
        Path to valid cached module, or None if not found/invalid
    """
    compiled_path = get_compiled_path(plugin_path, cache_dir)

    if is_cache_valid(plugin_path, compiled_path):
        return compiled_path

    return None
