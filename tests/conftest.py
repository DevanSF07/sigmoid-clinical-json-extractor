"""Keep tiny CPU smoke tests fast on many-core hosts."""
def pytest_sessionstart(session):
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
