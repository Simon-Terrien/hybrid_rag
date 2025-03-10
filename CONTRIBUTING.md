# Contributing Guidelines

Thank you for your interest in contributing to the Docling RAG System! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## How to Contribute

There are many ways to contribute to this project:

1. **Report bugs**: Submit bugs and issues on our issue tracker
2. **Suggest features**: Propose new features or improvements
3. **Improve documentation**: Fix typos, clarify language, add examples
4. **Submit code changes**: Implement new features or fix bugs
5. **Review code**: Review pull requests from other contributors

## Development Process

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/docling-rag.git`
3. Set up the development environment:
   ```bash
   cd docling-rag
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

### Code Style and Standards

We follow these coding standards:

- **PEP 8**: For Python code style
- **Type hints**: Use Python type hints for function signatures
- **Docstrings**: Use Google style docstrings for documentation
- **Tests**: Write tests for new functionality

We use the following tools:

- **Black**: For code formatting
- **isort**: For import sorting
- **mypy**: For type checking
- **flake8**: For linting
- **pytest**: For testing

Run pre-commit checks before submitting:

```bash
# Install pre-commit hooks
pre-commit install

# Run all checks
pre-commit run --all-files
```

### Git Workflow

1. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name`
2. Make your changes and commit them with clear, descriptive commit messages
3. Push your branch: `git push origin feature/your-feature-name`
4. Submit a pull request to the main repository

### Pull Request Process

1. Ensure your code passes all tests and linting checks
2. Update documentation if necessary
3. Include a clear description of the changes in your pull request
4. Link any related issues in your pull request description
5. Wait for review from maintainers

## Project Structure

```
docling-rag/
├── config.py                  # Configuration settings
├── doclingroc/                # Core RAG components
│   ├── docling_processor.py   # Document processing
│   └── docling_rag_orchestrator.py  # Main orchestrator
├── vectorproc/                # Vector processing
│   ├── vector_indexer.py      # Vector database management
│   └── semantic_search.py     # Semantic search
├── searchproc/                # Search processing
│   └── hybrid_search.py       # Hybrid search
├── fastapi_app.py             # FastAPI application
├── rag_orchestrator_api.py    # API orchestrator
├── main.py                    # CLI interface
├── server.py                  # Standalone server
└── utils.py                   # Utility functions
```

When adding new features, maintain this structure and follow existing patterns.

## Adding New Components

### New Document Processor

To add a new document processor:

1. Create a new class in `doclingroc/`
2. Implement required methods (`process_document`, `process_directory`)
3. Update the orchestrator to use your processor

### New Search Functionality

To add new search functionality:

1. Create a new search class in `searchproc/`
2. Implement the search interface (search method)
3. Update the orchestrator to use your search method

## Testing

Run tests with pytest:

```bash
pytest
```

When adding new features, add appropriate tests in the `tests/` directory:

- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows

## Documentation

Update documentation when changes affect user-facing functionality:

- Update relevant README sections
- Update docstrings for public APIs
- Add examples for new features
- Update configuration documentation

## Release Process

1. Version numbers follow [Semantic Versioning](https://semver.org/)
2. Maintain a changelog in CHANGELOG.md
3. Releases are tagged in git and published on GitHub Releases

## Questions and Discussions

For questions or discussions:

- Open a discussion on GitHub
- Reach out to maintainers
- Join our community chat

Thank you for contributing to the Docling RAG System!