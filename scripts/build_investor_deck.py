"""
Build a minimal investor deck PPTX without external dependencies.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape

OUTPUT_PATH = Path("reports/investor_deck.pptx")

SLIDES = [
    {
        "title": "MertFormer Titan",
        "bullets": [
            "Edge-native coding model (2.6B target)",
            "Operator-mode verified pipeline",
            "Offline-first, mobile compute focus",
        ],
    },
    {
        "title": "Problem",
        "bullets": [
            "Cloud AI is costly and slow for regulated workflows",
            "Privacy and data sovereignty are hard to guarantee",
            "Latency blocks real-time, on-device coding",
        ],
    },
    {
        "title": "Solution",
        "bullets": [
            "On-device inference with BitNet 1.58-bit quantization",
            "LiquidRouter MoE for stable, efficient routing",
            "Long-context attention optimized for code",
        ],
    },
    {
        "title": "Product",
        "bullets": [
            "2.6B parameter target model for coding",
            "Offline operation on mobile-class hardware",
            "Security-first with reproducibility gates",
        ],
    },
    {
        "title": "Architecture Highlights",
        "bullets": [
            "BitLinear + Liquid Neural Networks",
            "MLA attention with long-context readiness",
            "Sparse MoE with temporal routing",
        ],
    },
    {
        "title": "Safety & Reliability",
        "bullets": [
            "Kill-switch for non-finite stability events",
            "Failure budget and pivot triggers",
            "Checkpoint restore drills and reproducibility stamp",
        ],
    },
    {
        "title": "Evaluation Plan",
        "bullets": [
            "Golden sample suite (50 prompts)",
            "HumanEval/MBPP output generation",
            "Full 1MB overfit gate on training hardware",
        ],
    },
    {
        "title": "Market & Use Cases",
        "bullets": [
            "Regulated enterprises and defense workflows",
            "On-device copilots for low-connectivity regions",
            "Private inference stacks with data control",
        ],
    },
    {
        "title": "Roadmap",
        "bullets": [
            "Master run (2.6B) with telemetry gates",
            "Benchmark validation + pilot deployments",
            "Asset stack + demo video for launch",
        ],
    },
    {
        "title": "Ask",
        "bullets": [
            "Pilot partners for edge coding workflows",
            "Infrastructure credits for training runs",
            "Strategic advisors for deployment validation",
        ],
    },
]

NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def paragraph(text: str, size: int = 2400, bold: bool = False, bullet: bool = False) -> str:
    rpr = f"<a:rPr sz=\"{size}\"{(' b=\"1\"' if bold else '')}/>"
    bu = "<a:buChar char=\"•\"/>" if bullet else "<a:buNone/>"
    return (
        f"<a:p><a:pPr>{bu}</a:pPr>"
        f"<a:r>{rpr}<a:t>{escape(text)}</a:t></a:r><a:endParaRPr sz=\"{size}\"/>"
        f"</a:p>"
    )


def text_box(shape_id: int, name: str, x: int, y: int, cx: int, cy: int, paragraphs: str) -> str:
    return (
        f"<p:sp>"
        f"<p:nvSpPr><p:cNvPr id=\"{shape_id}\" name=\"{escape(name)}\"/>"
        f"<p:cNvSpPr txBox=\"1\"/><p:nvPr/></p:nvSpPr>"
        f"<p:spPr><a:xfrm><a:off x=\"{x}\" y=\"{y}\"/>"
        f"<a:ext cx=\"{cx}\" cy=\"{cy}\"/></a:xfrm>"
        f"<a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></p:spPr>"
        f"<p:txBody><a:bodyPr wrap=\"square\"/><a:lstStyle/>"
        f"{paragraphs}"
        f"</p:txBody>"
        f"</p:sp>"
    )


def slide_xml(title: str, bullets: list[str]) -> str:
    title_paragraphs = paragraph(title, size=3600, bold=True, bullet=False)
    body_paragraphs = "".join(paragraph(b, size=2400, bullet=True) for b in bullets)

    title_box = text_box(2, "Title", 457200, 228600, 8229600, 685800, title_paragraphs)
    body_box = text_box(3, "Body", 685800, 1143000, 7772400, 3429000, body_paragraphs)

    return (
        f"<p:sld xmlns:p=\"{NS_P}\" xmlns:a=\"{NS_A}\" xmlns:r=\"{NS_R}\">"
        f"<p:cSld><p:spTree>"
        f"<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/>"
        f"<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        f"<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/>"
        f"<a:ext cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>"
        f"{title_box}{body_box}"
        f"</p:spTree></p:cSld>"
        f"<p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
        f"</p:sld>"
    )


def write_pptx(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_types = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
        "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
        "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
        "<Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>",
        "<Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>",
        "<Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>",
        "<Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>",
        "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>",
        "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>",
    ]
    for idx in range(1, len(SLIDES) + 1):
        content_types.append(
            f"<Override PartName=\"/ppt/slides/slide{idx}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"
        )
    content_types.append("</Types>")

    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    slide_ids = []
    rels = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
        f"<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>",
    ]
    for idx in range(1, len(SLIDES) + 1):
        rels.append(
            f"<Relationship Id=\"rId{idx+1}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide{idx}.xml\"/>"
        )
        slide_ids.append((256 + idx - 1, f"rId{idx+1}"))
    rels.append("</Relationships>")

    presentation = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        f"<p:presentation xmlns:p=\"{NS_P}\" xmlns:a=\"{NS_A}\" xmlns:r=\"{NS_R}\">",
        "<p:sldMasterIdLst>",
        "<p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/>",
        "</p:sldMasterIdLst>",
        "<p:sldIdLst>",
    ]
    for slide_id, rel_id in slide_ids:
        presentation.append(f"<p:sldId id=\"{slide_id}\" r:id=\"{rel_id}\"/>")
    presentation.extend([
        "</p:sldIdLst>",
        "<p:sldSz cx=\"9144000\" cy=\"5143500\" type=\"screen16x9\"/>",
        "<p:notesSz cx=\"6858000\" cy=\"9144000\"/>",
        "</p:presentation>",
    ])

    slide_master = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sldMaster xmlns:p="{NS_P}" xmlns:a="{NS_A}" xmlns:r="{NS_R}">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst>
    <p:sldLayoutId id="2147483649" r:id="rId1"/>
  </p:sldLayoutIdLst>
</p:sldMaster>
"""

    slide_master_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""

    slide_layout = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sldLayout xmlns:p="{NS_P}" xmlns:a="{NS_A}" xmlns:r="{NS_R}" type="title" preserve="1">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr>
    <a:masterClrMapping/>
  </p:clrMapOvr>
</p:sldLayout>
"""

    slide_layout_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""

    theme = f"""<?xml version="1.0" encoding="UTF-8"?>
<a:theme xmlns:a="{NS_A}" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:sysClr val="windowText" lastClr="000000"/></a:dk1>
      <a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F1F1F"/></a:dk2>
      <a:lt2><a:srgbClr val="EEEEEE"/></a:lt2>
      <a:accent1><a:srgbClr val="4F81BD"/></a:accent1>
      <a:accent2><a:srgbClr val="C0504D"/></a:accent2>
      <a:accent3><a:srgbClr val="9BBB59"/></a:accent3>
      <a:accent4><a:srgbClr val="8064A2"/></a:accent4>
      <a:accent5><a:srgbClr val="4BACC6"/></a:accent5>
      <a:accent6><a:srgbClr val="F79646"/></a:accent6>
      <a:hlink><a:srgbClr val="0000FF"/></a:hlink>
      <a:folHlink><a:srgbClr val="800080"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office">
      <a:majorFont>
        <a:latin typeface="Calibri"/>
      </a:majorFont>
      <a:minorFont>
        <a:latin typeface="Calibri"/>
      </a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="Office">
      <a:fillStyleLst/>
      <a:lnStyleLst/>
      <a:effectStyleLst/>
      <a:bgFillStyleLst/>
    </a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>
"""

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = f"""<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>MertFormer Titan Investor Deck</dc:title>
  <dc:creator>MertFormer</dc:creator>
  <cp:lastModifiedBy>MertFormer</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    app = """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>MertFormer Deck Builder</Application>
</Properties>
"""

    with ZipFile(output_path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "\n".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("ppt/presentation.xml", "\n".join(presentation))
        zf.writestr("ppt/_rels/presentation.xml.rels", "\n".join(rels))
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master)
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels)
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout)
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels)
        zf.writestr("ppt/theme/theme1.xml", theme)
        zf.writestr("docProps/core.xml", core)
        zf.writestr("docProps/app.xml", app)

        for idx, slide in enumerate(SLIDES, 1):
            zf.writestr(f"ppt/slides/slide{idx}.xml", slide_xml(slide["title"], slide["bullets"]))
            zf.writestr(
                f"ppt/slides/_rels/slide{idx}.xml.rels",
                """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
  <Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>
</Relationships>
""",
            )


if __name__ == "__main__":
    write_pptx(OUTPUT_PATH)
    print(f"Deck written to {OUTPUT_PATH}")
