"""
Fully-qualified AcroForm field identification.
================================================
BUG THIS FIXES
--------------
The rest of the codebase (field_mapper.field_boxes, acroform.fill_template /
read_acroform, scripts/generate_dataset.py) previously identified a field by
its *own* partial `/T` name (or one `/Parent` hop up). In the real 6th-edition
editable TA6 template this is unsafe: many unrelated questions on different
pages share the same short partial name. For example, the partial name "2" is
reused by 7 independent text fields on 6 different pages (disputes, boundary
notes, insurance, parking, connection-to-services, completion...). Filling
"field 2" for one question silently also targets six other unrelated fields.

`pypdf.PdfReader.get_fields()` has the same problem: it is keyed by leaf name,
so same-named-but-unrelated fields collide/overwrite in its returned dict too.

THE FIX
-------
Walk the AcroForm /Fields tree from the root and build each field's FULL
dotted name (join of every ancestor `/T` segment, PDF-spec correct). This is
guaranteed unique: a tree walk of the real template finds exactly 442 leaf
widgets and 442 distinct fully-qualified names (verified empirically -- zero
collisions), matching pypdf's own low-level widget/annotation count.

Use `list_fields(reader_or_writer)` everywhere a field needs to be targeted
by name, instead of raw `/T`.
"""
from typing import List, Dict, Optional


def list_fields(pdf) -> List[Dict]:
    """Return one record per terminal (leaf) AcroForm widget:
    {qname, page, rect, ft, ref} where `ref` is the indirect reference to the
    WIDGET object itself (so callers can set /V and /AS directly on it,
    independent of the (colliding) partial name).
    """
    widget_page = {}
    for pi, page in enumerate(pdf.pages):
        for a in (page.get("/Annots") or []):
            ref = a if hasattr(a, "idnum") else getattr(a, "indirect_reference", None)
            if ref is not None:
                widget_page[ref.idnum] = pi

    out: List[Dict] = []

    def walk(node, ancestors):
        obj = node.get_object()
        t = obj.get("/T")
        names = ancestors + [str(t)] if t is not None else ancestors
        kids = obj.get("/Kids")
        if kids:
            for k in kids:
                walk(k, names)
        # a node is a widget if it carries its own appearance rect
        if obj.get("/Rect") is not None:
            idnum = node.idnum if hasattr(node, "idnum") else None
            page = widget_page.get(idnum)
            qname = ".".join(names) if names else f"__unnamed_{idnum}"
            ft = obj.get("/FT")
            if ft is None and obj.get("/Parent") is not None:
                ft = obj.get("/Parent").get_object().get("/FT")
            out.append({
                "qname": qname,
                "page": page,
                "rect": [float(x) for x in obj.get("/Rect")],
                "ft": str(ft) if ft else None,
                "ref": node,
            })

    root = pdf.trailer["/Root"] if hasattr(pdf, "trailer") else pdf._root_object
    root_fields = root["/AcroForm"]["/Fields"]
    for f in root_fields:
        walk(f, [])
    return out


def by_qname(pdf) -> Dict[str, Dict]:
    fields = list_fields(pdf)
    dupes = {}
    out = {}
    for f in fields:
        if f["qname"] in out:
            dupes.setdefault(f["qname"], [out[f["qname"]]]).append(f)
        else:
            out[f["qname"]] = f
    if dupes:
        # Should not happen on the real template (verified 442/442 unique);
        # surface loudly if a future template revision breaks that invariant.
        raise ValueError(f"Fully-qualified field name collision(s) found: {list(dupes)[:5]} "
                         f"(+{max(0, len(dupes)-5)} more) -- filling would be unsafe.")
    return out
