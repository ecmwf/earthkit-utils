#!/usr/bin/env python3

# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

# The tests run against ``earthkit.utils.html``. During the move they were
# validated against the original implementations in earthkit-data and earthkit-geo.
html = pytest.importorskip("earthkit.utils.html")


def test_css_table():
    out = html.css("table")
    assert out.startswith('<style type="text/css">')
    assert out.endswith("</style>")
    assert "table.ek td" in out


def test_css_unknown():
    with pytest.raises(FileNotFoundError):
        html.css("does-not-exist")
