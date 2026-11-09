# (C) Copyright 2025 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Helpers to build the HTML representation of objects."""

from importlib import resources


def css(name):
    path = resources.files("earthkit.utils") / "static" / f"{name}.css"
    return """<style type="text/css">%s</style>""" % (path.read_text(),)
