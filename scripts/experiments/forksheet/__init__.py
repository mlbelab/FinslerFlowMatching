"""The ForkSheet: one bifurcating cloud on a curved sheet, and its drawing helpers.

A second route-selection cloud, built to ask a question the Sheet cannot.  The Sheet has
one destination and the only failure is *where* mass crosses; here there are two, and an
arm can hit both marginals while sending the wrong cells to the wrong fate.  How much mass
each destination carries is a knob (``shares``), defaulting to an even split, so "did each
cell reach its own arm" can be asked at a proportion an even split cannot fake.

Everything downstream is the Sheet's: :mod:`scripts.experiments.sheet.split`,
:mod:`~scripts.experiments.sheet.protocol` and :mod:`~scripts.experiments.sheet.evaluate`
are written against a :class:`~scripts.experiments.sheet.datasets.RouteDataset` and not
against the Sheet, so this package adds a cloud and a wireframe and nothing else.  The
Sheet's own registry stays pinned to one name, which is why this cloud lives beside it
rather than in it.
"""
