import sys

# Check if pytest is running
is_pytest_running = "pytest" in sys.modules

if not is_pytest_running:
    # Raise an ImportError if an attempt is made to import a test module, but only if not running under pytest
    if any(mod_name.startswith('test.') for mod_name in sys.modules):
        raise ImportError("Importing from test modules is not allowed.")

    from .pseudodigraph import PseudoDiGraph
    from .planarpdgraph import PlanarPDGraph

    # Any other necessary imports
else:
    # Allow importing test modules when running under pytest
    pass
