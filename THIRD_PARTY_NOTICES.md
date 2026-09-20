# Third-party notices

This file records every piece of third-party code vendored into this repository, the upstream
project it came from, the exact tag and commit it was taken at, and its licence. Plan §10.7
requires it to be updated in the **same commit** that vendors the code.

| Component | Upstream | Tag / commit | Licence | Vendored into |
|---|---|---|---|---|
| openzim-mcp | https://github.com/cameronrye/openzim-mcp | v3.3.4 / 9358db06f205bb0b95cc938c68405851a0e205a8 | MIT | `tutor/retrieval/zim/archive.py` (from `openzim_mcp/zim/archive.py`: `ZIM_MAGIC`, `_declared_zim_size`, `has_zim_signature`, `is_truncated_zim`, vendored near-verbatim); `tutor/retrieval/zim/resolve.py` (from `openzim_mcp/zim/redirects.py`: `resolve_redirect_chain`, adapted into `resolve_entry`); `tutor/retrieval/zim/content.py` (from `openzim_mcp/content_processor.py`: only the `select_main_content` landmark-priority idea kept, reimplemented; the html2text/heading-locator rendering pipeline was replaced with an online single-pass renderer); `tutor/retrieval/zim/bundle.py` (from `openzim_mcp/bundle.py`: only the "one parse, one bundle" shape of `extract_entry_bundle` kept; `_compute_section_offsets`/`_locate_heading_text`/`_match_decorated_heading_line`/`_strip_md_inline_decorations` were dropped in favour of the online offsets `content.py` records, and infobox/link extraction were rewritten from scratch) |

### openzim-mcp MIT licence text

```
MIT License

Copyright (c) 2025-2026 Cameron Rye

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtime dependencies (not vendored)

| Component | Source | Licence | Notes |
|---|---|---|---|
| libzim (`python-libzim`) | PyPI, installed as a regular dependency (not vendored) | GPL-3.0-or-later. Exact installed package metadata (`python -m pip show libzim`, version 3.13.0): `License-Expression: GPL-3.0-or-later`, `Home-page: https://github.com/openzim/python-libzim`. | Installed from PyPI, no code copied into this repo. **Flag for the project owner**: distributing a bundle that includes this dependency (e.g. a packaged installer) has GPL-3.0-or-later implications that must be reviewed at the installer milestone, before any such bundle ships. |
| beautifulsoup4 | PyPI, installed as a regular dependency (not vendored) | MIT | No code copied into this repo. |

## Design references — no code copied

| Project | Why it is here | Rule |
|---|---|---|
| Zim-Indexer | Ranking and indexing ideas only. No licence was found upstream (plan §0.5). | Design reference. No copied code, ever. |
