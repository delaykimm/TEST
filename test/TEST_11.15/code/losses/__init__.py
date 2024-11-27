# import pkgutil

# __all__ = []
# for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
#     __all__.append(module_name)
#     module = loader.find_module(module_name).load_module(module_name)
#     exec('%s = module' % module_name)

import importlib
import pkgutil

__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    # 모듈 로드
    module = importlib.import_module(f'{__name__}.{module_name}')
    # 전역 네임스페이스에 동적으로 모듈 추가
    globals()[module_name] = module