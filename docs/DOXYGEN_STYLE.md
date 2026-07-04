# Doxygen Comment Style

The source code uses lightweight Doxygen-style docstrings. I used this style so the functions are easy to scan while still being compatible with a Doxygen documentation pass.

Typical module comment:

```python
"""@file module.py
@brief Short description of what this module contains.
"""
```

Typical function comment:

```python
def function_name(arg1, arg2):
    """@brief Explain the function in one sentence.

    @param arg1 Explain the first argument.
    @param arg2 Explain the second argument.
    @return Explain the returned value.
    """
```

Tags used in this project:

| Tag | Meaning |
|---|---|
| `@file` | File-level description |
| `@brief` | Short summary |
| `@param` | Parameter description |
| `@return` | Return value description |

The comments are intentionally simple. The goal is not to over-document obvious Python syntax, but to make the IEPS, SCF, geometry, and evaluation code easier to follow.
