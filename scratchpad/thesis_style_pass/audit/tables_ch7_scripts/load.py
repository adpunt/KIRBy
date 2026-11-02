import importlib.util, sys
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('c7', '/Users/apunt/repos/KIRBy/tests/analysis/ch7_numbers.py')
c7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c7)
c7.CTX = c7.Ctx('/Volumes/seagate/kirby_results_2026-10-10', '/Volumes/seagate/kirby_results_2026-10-09_noise_view')
