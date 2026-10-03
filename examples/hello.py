"""The smallest call.  Run from the repo root:  python3 -m examples.hello"""

import thunc

thunc.configure(backend="claude-code")
print(thunc.call("Say hello in five words or fewer."))
