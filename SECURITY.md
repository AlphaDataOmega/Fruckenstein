# Security and safety

- The Windows executable is not code-signed. Verify its SHA-256 before running it.
- The build is a local MuJoCo simulation and does not require network access.
- Learned memory is written beside the executable in `RelationalDuckGroupData/`.
- Do not connect this research controller to physical actuators without independent limits, emergency stop, isolation, and hardware validation.
- Report suspected security problems privately through the GitHub repository owner rather than posting exploit details publicly.
