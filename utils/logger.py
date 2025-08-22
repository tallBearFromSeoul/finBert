from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging
import multiprocessing
import sys
import time

class Logger:
    _logger: logging.Logger = logging.getLogger("app")
    _file_handler: Optional[logging.Handler] = None
    _configured: bool = False

    class UTCFormatter(logging.Formatter):
        converter = time.gmtime
        _fmt = "%(asctime)sZ - %(levelname)s - %(message)s"
        _datefmt = "%Y-%m-%dT%H:%M:%S"

        def __init__(self) -> None:
            super().__init__(Logger.UTCFormatter._fmt, Logger.UTCFormatter._datefmt)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _get_entry_name(cls) -> str:
        # Try __main__.__file__, else sys.argv[0], else default to "app"
        import __main__ as _m

        fp = getattr(_m, "__file__", None)
        if fp:
            return Path(fp).stem
        if len(sys.argv) > 0 and sys.argv[0]:
            return Path(sys.argv[0]).stem
        return "app"

    @classmethod
    def _get_context(cls) -> str:
        try:
            # Caller: 0=_get_context, 1=Logger.info, 2=actual caller
            frame = sys._getframe(2)
            func = frame.f_code.co_name
            locals_ = frame.f_locals
            if "self" in locals_:
                clsname = type(locals_["self"]).__name__
                return f"[{clsname}::{func}]"
            elif "cls" in locals_:
                clsname = locals_["cls"].__name__
                return f"[{clsname}::{func}]"
            else:
                return f"[{func}]"
        except Exception:
            return "[unknown]"

    @classmethod
    def _ensure_configured(cls) -> None:
        if cls._configured:
            return

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(cls.UTCFormatter())
        cls._logger.addHandler(console)

        cls._logger.setLevel(logging.INFO)
        cls._logger.propagate = False
        cls._configured = True

    @classmethod
    def setup_file(cls, path: str) -> None:
        path_obj = Path(path)
        entry = cls._get_entry_name()
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_name = multiprocessing.current_process().name
        filename = f"{entry}_{date_str}_{process_name}.log"
        file_path = path_obj / filename

        path_obj.mkdir(parents=True, exist_ok=True)

        if cls._file_handler is not None:
            cls._logger.removeHandler(cls._file_handler)

        cls._file_handler = logging.FileHandler(file_path, encoding="utf-8")
        cls._file_handler.setLevel(logging.INFO)
        cls._file_handler.setFormatter(cls.UTCFormatter())
        cls._logger.addHandler(cls._file_handler)

        print(f"[Logger] writing to {file_path!r}")

    @classmethod
    def info(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._ensure_configured()
        context = cls._get_context()
        cls._logger.info(f"{context} {msg}", *args, **kwargs)

    @classmethod
    def debug(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._ensure_configured()
        context = cls._get_context()
        cls._logger.debug(f"{context} {msg}", *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._ensure_configured()
        context = cls._get_context()
        cls._logger.warning(f"{context} {msg}", *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._ensure_configured()
        context = cls._get_context()
        cls._logger.error(f"{context} {msg}", *args, **kwargs)

    @classmethod
    def fatal(cls, msg: str, *args: Any, **kwargs: Any) -> None:
        cls._ensure_configured()
        context = cls._get_context()
        cls._logger.fatal(f"{context} {msg}", *args, **kwargs)
