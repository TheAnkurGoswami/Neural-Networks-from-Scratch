import pytest
import logging
import os


@pytest.fixture(autouse=True)
def setup_logging(request):
    """Fixture to configure logging for each test case."""
    test_name = request.node.originalname  # Test function name
    param_values = (
        request.node.callspec.params
        if hasattr(request.node, "callspec")
        else {}
    )

    # Convert parameter values into a filename-friendly format
    param_str = "_".join(f"{k}-{v}" for k, v in param_values.items())
    log_filename = os.path.join(
        "tests",
        "logs",
        test_name,
        f"{param_str if param_str else 'no_param'}.log",
    )

    # Ensure the logs directory exists
    os.makedirs(os.path.join("tests", "logs", test_name), exist_ok=True)

    # Configure logging
    logger = logging.getLogger()
    logger.handlers.clear()  # Remove old handlers to avoid duplicate logs
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode="w")
    msg_fmt = "%(asctime)s - %(levelname)s - %(message)s"
    msg_fmt = "%(message)s"
    formatter = logging.Formatter(msg_fmt)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    yield  # Let the test run
    # Cleanup: Remove handler after test completes
    logger.removeHandler(file_handler)
    file_handler.close()
